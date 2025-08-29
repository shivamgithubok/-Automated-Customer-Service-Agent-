import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
import hashlib
import os
from pathlib import Path


class Scraper:
    def __init__(self):
        # Setup FAISS path (same as RAGAgent)
        self.base_path = Path(__file__).parent.parent
        self.faiss_path = self.base_path / "vectorstore"
        self.faiss_path.parent.mkdir(parents=True, exist_ok=True)

        # Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Load or create FAISS store
        if self.faiss_path.exists():
            self.vector_store = FAISS.load_local(
                folder_path=str(self.faiss_path),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = FAISS.from_texts(
                texts=["Initial setup"],
                embedding=self.embeddings
            )
            self.vector_store.save_local(str(self.faiss_path))

        self.document_ids = {}  # maps content_hash -> url

    def save_vectorstore(self):
        """Persist FAISS index"""
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        self.faiss_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.faiss_path))

    def generate_hash(self, text: str) -> str:
        """Generate unique hash for deduplication"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ------------------------
    # 1. Scrape website and store
    # ------------------------
    def scrape_website(self, url: str) -> dict:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove unwanted tags
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        text = soup.get_text(separator=" ")
        clean_text = " ".join(text.split())

        # Generate unique hash
        doc_hash = self.generate_hash(clean_text)
        if doc_hash in self.document_ids:
            return {
                "status": "success",
                "message": "URL already processed",
                "document_id": self.document_ids[doc_hash]
            }

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_text(clean_text)
        docs = [Document(page_content=chunk, metadata={"source": url}) for chunk in texts]

        # Add to FAISS
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vector_store.add_documents(docs)

        # Save
        self.document_ids[doc_hash] = url
        self.save_vectorstore()

        return {
            "status": "success",
            "chunks_processed": len(docs),
            "document_id": url
        }

    # ------------------------
    # 2. LLM Setup
    # ------------------------
    def get_llm(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0,
        )

    # ------------------------
    # 3. Ask question (retrieval)
    # ------------------------
    def ask_question(self, question: str, k: int = 3) -> str:
        if not self.vector_store:
            raise ValueError("❌ No data indexed yet. Scrape a website first.")

        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in docs])

        llm = self.get_llm()

        prompt = ChatPromptTemplate.from_messages([
                ("system", 
                "You are a precise assistant. Use ONLY the provided context to answer. "
                "If the answer is not explicitly in the context, reply with: 'Not found in provided sources.'\n\n"
                "When answering, cite the relevant source section (e.g., heading, title, or file name) "
                "so the user knows where it came from."
                ),
                ("human", 
                "Context (from HTML source):\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer format:\n"
                "- Direct answer\n"
                "- Cite the exact section or snippet from context where answer was found"
                )
            ])

        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})

        return response.content
