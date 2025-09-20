import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import hashlib
import os
from pathlib import Path


class Scraper:
    def __init__(self):
        # Setup Chroma path for scraped content
        self.base_path = Path(__file__).parent.parent
        self.chroma_path = self.base_path / "chromadb_scraping"
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        # Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Initialize Chroma store
        self.vector_store = Chroma(
            persist_directory=str(self.chroma_path),
            embedding_function=self.embeddings,
            collection_name="scraped_content"
        )

        self.document_ids = {}  # maps content_hash -> url

    def persist_vectorstore(self):
        """Persist ChromaDB store"""
        if self.vector_store is None:
            raise ValueError("No vector store to persist")
        self.vector_store.persist()

    def generate_hash(self, text: str) -> str:
        """Generate unique hash for deduplication"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ------------------------
    # 1. Scrape website and store
    # ------------------------
    def scrape_website(self, url: str) -> dict:
        """Scrape website content and store in vector database"""
        try:
            # Validate URL
            if not url:
                raise ValueError("URL cannot be empty")
                
            # Add headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Fetch the webpage
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            # Ensure we're getting HTML content
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                raise ValueError(f"Invalid content type: {content_type}")

            # Parse HTML
            soup = BeautifulSoup(response.content, "html.parser")

            # Remove unwanted elements
            for element in soup(['script', 'style', 'head', 'title', 'meta', '[document]', 'header', 'footer', 'nav']):
                element.decompose()

            # Extract text with better formatting
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean the text
            clean_text = ' '.join(line.strip() for line in text.splitlines() if line.strip())
            
            if not clean_text:
                raise ValueError("No content found on the webpage")
                
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch webpage: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error scraping webpage: {str(e)}")

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
        docs = [Document(page_content=chunk, metadata={
            "source": url,
            "doc_hash": doc_hash,
            "chunk_id": f"{doc_hash}_{i}"
        }) for i, chunk in enumerate(texts)]

        # Add to ChromaDB
        self.vector_store.add_documents(docs)
        
        # Persist changes
        self.document_ids[doc_hash] = url
        self.persist_vectorstore()
        print(docs)
        return {
            "status": "success",
            "chunks_processed": len(docs),
            "document_id": url,
            "collection": "scraped_content"
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

        if not self.vector_store:
            raise ValueError("❌ No data indexed yet. Scrape a website first.")

        # Use direct similarity search without search_type parameter
        docs = self.vector_store.similarity_search(
            query=question,
            k=k
        )
        
        # Join document content with source information
        context = "\n\n".join([
            f"{d.page_content}\nSource: {d.metadata.get('source', 'Unknown')}"
            for d in docs
        ])

        llm = self.get_llm()

        prompt = ChatPromptTemplate.from_messages([
            ("system", 
            "You are a precise assistant. Use ONLY the provided context to answer. "
            "If the answer is not explicitly in the context, reply with: 'Not found in provided sources.'\n\n"
            "When answering, cite the relevant source section (e.g., heading, title, or file name) "
            "so the user knows where it came from.\n\n"
            "If the user mentions specific columns and scraping criteria (such as class or ID names), "
            "scrape the provided HTML context for the relevant data and organize it into the requested table format.\n"
            "For example, if the user provides class names 'container' and 'fruit-price', "
            "scrape the data from the 'container' class for the fruit names and from 'fruit-price' for the prices.\n\n"
            "Example:\n"
            "User query: 'Scrape the fruit names from the 'container' class and the prices from the 'fruit-price' class. Present them in a table.'\n"
            "Assistant response: \n"
            "| Fruit Name  | Price  |\n"
            "|-------------|--------|\n"
            "| Apple       | $2.99  |\n"
            "| Banana      | $1.49  |\n"
            "| Cherry      | $3.99  |\n"
            "*Source: Scraped from the 'container' and 'fruit-price' classes*\n"
            ),
            ("human", 
            "Context (from HTML source):\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer format:\n"
            "- Direct answer\n"
            "- Cite the exact section or snippet from context where answer was found\n"
            "- If scraping is requested, organize the data into columns and format it as a markdown table"
            )
        ])


        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})

        return response.content
