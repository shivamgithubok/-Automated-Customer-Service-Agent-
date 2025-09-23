import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from typing import List, Dict
from prompts.formate import FormatPrompt  # Import the format prompt

# Load environment variables
load_dotenv()

class RAGAgent:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.faiss_path = self.base_path / "production_vectorstore"  # Use a new store
        self.docstore_path = self.base_path / "production_docstore"
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        self.vectorstore = None
        self.store = InMemoryStore()
        self.formating_prompt = FormatPrompt()  # Initialize FormatPrompt instance

        self.retriever = None  # Will be initialized in `load_or_create_retriever`

        # Cross-Encoder for Re-ranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

        # Powerful LLM and Output Parser
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.output_parser = StrOutputParser()

        # Advanced Prompt Engineering (Persona + Chain-of-Thought)
        self.prompt_template = self._create_prompt_template()

        # Initialize the retriever and vector store
        self.load_or_create_retriever()

    def _create_prompt_template(self) -> PromptTemplate:
        """
        --- The Production-Grade General Assistant Prompt ---
        This prompt instructs the LLM to act as an expert synthesizer of information,
        focusing on user intent, logical reasoning, and providing comprehensive, trustworthy answers.
        """
        
        # Example context data, can be dynamic from the user input or document processing
        context_data = {
            'projects': ["Project Alpha", "Project Beta", "Project Gamma", "Project Delta"],
            'formula': "E = mc^2"
        }

        # 1. Generate the formatted context using FormatPrompt
        formatted_context = self.formating_prompt.generate_format_prompt(context_data['projects'], context_data['formula'])

        # 2. Construct the final prompt template
        template = f"""
        You are an expert Enterprise Knowledge Assistant. Your purpose is to help company employees find information and understand complex documents quickly and accurately. You are not just a search engine; you are a reasoning and synthesis engine.

        **YOUR CORE DIRECTIVE:**
        Your primary goal is to synthesize the information in the provided `CONTEXT FROM DOCUMENTS` to give a comprehensive, logical, and easy-to-understand answer to the user's `QUESTION`.

        **HERE IS YOUR STEP-BY-STEP THINKING PROCESS:**

        1.  **Deconstruct the User's Question:** First, analyze the `QUESTION` and the `CHAT HISTORY`. What is the user's *real* intent? Are they asking for a specific fact, a summary, a comparison, or a step-by-step process?

        2.  **Scour the Context:** Read through all the provided `CONTEXT FROM DOCUMENTS`. Identify all relevant paragraphs, sentences, data points, and examples that relate to the user's intent.

        3.  **Synthesize, Don't Just Recite:** Do NOT just copy and paste chunks from the context. Your value is in your ability to connect the dots. Synthesize the key points from the different parts of the context into a single, coherent, and logical answer. If the context provides a process, lay it out in clear steps. If it provides data, summarize the key findings.

        4.  **Answer Completely and Confidently:** Formulate your final answer. Start with a direct response, then provide the supporting details or examples you synthesized from the context. Write in a clear, professional, and helpful tone.

        **CRITICAL RULES FOR TRUST AND ACCURACY:**

        *   **GROUNDING IS EVERYTHING:** You MUST base your entire answer *strictly* on the information found within the `CONTEXT FROM DOCUMENTS`.
        *   **HANDLE MISSING INFORMATION:** If the context does not contain the information needed to answer the question, you must clearly state that. For example, say: "Based on the documents provided, I could not find information regarding [the user's topic]." **DO NOT, under any circumstances, make up information or use external knowledge.**
        *   **HANDLE AMBIGUITY:** If the user's question is vague, ask for clarification. For example: "To give you the most accurate answer, could you please specify which financial quarter you are interested in?"

        ---
        **CHAT HISTORY (for conversational context):**
        {{"chat_history"}}  # You can replace this with actual chat history data if available
        ---
        **CONTEXT FROM DOCUMENTS (your source of truth):**
        {formatted_context}  # This will insert the formatted context (lists, tables, formulas)
        ---
        **QUESTION (the user's request):**
        {{"question"}}  # Insert the question here
        ---
        **SYNTHESIZED ANSWER:**
        """
        return PromptTemplate.from_template(template)

    def load_or_create_retriever(self):
        """Initializes the entire retrieval system (vector store, doc store, and retriever)."""
        if self.faiss_path.exists():
            print(f"Loading existing vector store from {self.faiss_path}...")
            self.vectorstore = FAISS.load_local(
                folder_path=str(self.faiss_path),
                embeddings=self.embeddings_model,
                allow_dangerous_deserialization=True
            )
        else:
            print("Creating a new, empty vector store.")
            # Create a dummy index to start with
            dummy_texts = ["initialization text"]
            self.vectorstore = FAISS.from_texts(texts=dummy_texts, embedding=self.embeddings_model)
            self.vectorstore.delete(self.vectorstore.index_to_docstore_id.values())  # Clear dummy text
            self.vectorstore.save_local(str(self.faiss_path))

        # The ParentDocumentRetriever orchestrates the search-and-retrieve process
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )
        print("Retriever is ready.")

    def process_documents(self, file_paths: List[str]):
        """Process a list of files using Unstructured.io and add them to the retriever's stores."""
        for file_path in file_paths:
            try:
                print(f"Processing document: {file_path}")
                loader = UnstructuredFileLoader(file_path)
                docs = loader.load()
                self.retriever.add_documents(docs, ids=None)
                self.vectorstore.save_local(str(self.faiss_path))
                print(f"Successfully processed and stored {file_path}")
            except Exception as e:
                print(f"Failed to process {file_path}. Error: {e}")

    def _get_and_rerank_documents(self, question: str) -> List[Dict]:
        """Retrieve and re-rank documents based on relevance."""
        print("Retrieving initial documents...")
        sub_docs = self.vectorstore.similarity_search(question, k=20)
        retrieved_docs = self.retriever.invoke(question)

        if not retrieved_docs:
            return []

        print(f"Re-ranking {len(retrieved_docs)} documents...")
        pairs = [[question, doc.page_content] for doc in retrieved_docs]
        scores = self.reranker.predict(pairs)

        scored_docs = list(zip(scores, retrieved_docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        top_docs = []
        for score, doc in scored_docs[:4]:
            top_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })

        print(f"Selected {len(top_docs)} documents after re-ranking.")
        return top_docs

    def query(self, question: str, chat_history: List[Dict[str, str]]) -> str:
        """Query method that invokes the whole process."""
        reranked_docs = self._get_and_rerank_documents(question)
        context_text = "\n\n---\n\n".join([doc['content'] for doc in reranked_docs])

        formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

        chain = (
            {
                "context": lambda x: context_text, 
                "question": RunnablePassthrough(), 
                "chat_history": lambda x: formatted_history, 
            }
            | self.prompt_template
            | self.llm
            | self.output_parser
        )

        print("Invoking LLM chain to generate answer...")
        answer = chain.invoke(question)
        return answer