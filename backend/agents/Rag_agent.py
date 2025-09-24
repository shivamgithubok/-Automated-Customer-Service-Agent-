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

# Load environment variables
load_dotenv()

class RAGAgent:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.faiss_path = self.base_path / "production_vectorstore" # Use a new store
        self.docstore_path = self.base_path / "production_docstore"
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        self.vectorstore = None 
        self.store = InMemoryStore()

        self.retriever = None # Will be initialized in `load_or_create_retriever`

        # 3. Cross-Encoder for Re-ranking
        #    This model is much better at determining relevance than the initial search.
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

        # --- Part 3: A Better Brain ---
        
        # 1. Powerful LLM and Output Parser
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.output_parser = StrOutputParser()

        # 2. Advanced Prompt Engineering (Persona + Chain-of-Thought)
        self.prompt_template = self._create_prompt_template()

        # Initialize the retriever and vector store
        self.load_or_create_retriever()

    def _create_prompt_template(self) -> PromptTemplate:
        """
        --- The Production-Grade General Assistant Prompt ---
        This prompt instructs the LLM to act as an expert synthesizer of information,
        focusing on user intent, logical reasoning, and providing comprehensive, trustworthy answers.
        """
        template = """
        You are an expert Enterprise Knowledge Assistant. Your purpose is to help company employees find information and understand complex documents quickly and accurately. You are not just a search engine; you are a reasoning and synthesis engine.

        **YOUR CORE DIRECTIVE:**
        Your primary goal is to synthesize the information in the provided `CONTEXT FROM DOCUMENTS` to give a comprehensive, logical, and easy-to-understand answer to the user's `QUESTION`.

        **HERE IS YOUR STEP-BY-STEP THINKING PROCESS:**

        1.  **Deconstruct the User's Question:** First, analyze the `QUESTION` and the `CHAT HISTORY`. What is the user's *real* intent? Are they asking for a specific fact, a summary, a comparison, or a step-by-step process?

        2.  **Scour the Context:** Read through all the provided `CONTEXT FROM DOCUMENTS`. Identify all relevant paragraphs, sentences, data points, and examples that relate to the user's intent.

        3.  **Synthesize, Don't Just Recite:** Do NOT just copy and paste chunks from the context. Your value is in your ability to connect the dots. Synthesize the key points from the different parts of the context into a single, coherent, and logical answer. If the context provides a process, lay it out in clear steps. If it provides data, summarize the key findings.

        4.  **Answer Completely and Confidently:** Formulate your final answer. Start with a direct response, then provide the supporting details or examples you synthesized from the context. Write in a clear, professional, and helpful tone.

        

        ---
        **FORMATTING AND PRESENTATION:**
        You MUST format your output according to the user's implied need.

        *   **Use a NUMBERED LIST when:**
            - The user asks for "steps," a "process," or a "ranking."
            - The order of the items is important.
            - **Example Question:** "What are the steps to set up the MCP server?"
            - **Example Answer:**
                1) Install all necessary dependencies.
                2) Configure the environment variables.
                3) Start the server using the main script.
        
        
        *   **Use a MARKDOWN TABLE when:**
            - The user asks to "compare," "list details," or for information that has clear pairs of data (e.g., project and its description, term and its definition).
            - You need to present structured data clearly.
            - **Example Question:** "List the different types of Pokémon stats and their purpose."
            - **Example Answer:**
                ```markdown
                | Stat Name        | Purpose                               |
                |------------------|---------------------------------------|
                | HP               | Determines the Pokémon's health       |
                | Attack           | Affects damage of physical moves      |
                | Defense          | Reduces damage from physical moves    |
                ```

        *   **Use a FORMULA FORMAT (LaTeX) when:**
            - The user asks for a formula, equation, or a mathematical calculation.
            - **Example Question:** "What is the formula for calculating damage?"
            - **Example Answer:**
                The formula for calculating damage is often complex, but a simplified version is:
                `Damage = (((2 * Level / 5 + 2) * Power * A / D) / 50 + 2) * Modifier`
                Where:
                - `A` is the attacker's Attack or Special Attack stat.
                - `D` is the defender's Defense or Special Defense stat.
                - `Power` is the base power of the move.

        *   **For all other questions, use standard paragraphs and bullet points for clarity.**

        ---
        **CRITICAL RULES FOR TRUST AND ACCURACY:**

        *   **GROUNDING IS EVERYTHING:** You MUST base your entire answer *strictly* on the information found within the `CONTEXT FROM DOCUMENTS`.
        *   **HANDLE MISSING INFORMATION:** If the context does not contain the information needed to answer the question, you must clearly state that. For example, say: "Based on the documents provided, I could not find information regarding [the user's topic]." **DO NOT, under any circumstances, make up information or use external knowledge.**
        *   **HANDLE AMBIGUITY:** If the user's question is vague, ask for clarification. For example: "To give you the most accurate answer, could you please specify which financial quarter you are interested in?"
        * **MISSING INFORMATION:** If the context does not contain the answer, state that clearly. DO NOT make up information.
        ---
        **CHAT HISTORY (for conversational context):**
        {chat_history}
        ---
        **CONTEXT FROM DOCUMENTS (your source of truth):**
        {context}
        ---
        **QUESTION (the user's request):**
        {question}
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
            self.vectorstore.delete(self.vectorstore.index_to_docstore_id.values()) # Clear dummy text
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
        """
        Processes a list of files using Unstructured.io, splits them into parent/child chunks,
        and adds them to the retriever's stores.
        """
        for file_path in file_paths:
            try:
                print(f"Processing document: {file_path}")
                # UnstructuredFileLoader can handle .pdf, .docx, .txt, .pptx, and more.
                loader = UnstructuredFileLoader(file_path)
                docs = loader.load()
                
                # This single call handles splitting, embedding, and storing both
                # parent and child documents.
                self.retriever.add_documents(docs, ids=None)
                
                # Persist the updated vector store to disk
                self.vectorstore.save_local(str(self.faiss_path))
                print(f"Successfully processed and stored {file_path}")
            except Exception as e:
                print(f"Failed to process {file_path}. Error: {e}")

    def _get_and_rerank_documents(self, question: str) -> List[Dict]:
        """
        Retrieves initial documents and then uses a CrossEncoder to re-rank them
        for higher relevance before passing them to the LLM.
        """
        print("Retrieving initial documents...")
        # 1. Retrieve initial documents using the ParentDocumentRetriever
        sub_docs = self.vectorstore.similarity_search(question, k=20)
        retrieved_docs = self.retriever.invoke(question) # Gets the parent docs

        if not retrieved_docs:
            return []

        # 2. Re-rank the retrieved documents
        print(f"Re-ranking {len(retrieved_docs)} documents...")
        pairs = [[question, doc.page_content] for doc in retrieved_docs]
        scores = self.reranker.predict(pairs)
        
        # Combine docs with their scores and sort
        scored_docs = list(zip(scores, retrieved_docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Select the top 4 most relevant documents
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
        """
        The main method to ask a question. It orchestrates retrieval, re-ranking,
        prompting, and generation, all while being aware of the conversation history.
        """
        # 1. Retrieve and re-rank documents
        reranked_docs = self._get_and_rerank_documents(question)
        context_text = "\n\n---\n\n".join([doc['content'] for doc in reranked_docs])

        # 2. Format chat history for the prompt
        formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

        # 3. Create the LangChain Expression Language (LCEL) chain
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

        # 4. Invoke the chain to get the answer
        print("Invoking LLM chain to generate answer...")
        answer = chain.invoke(question)
        return answer