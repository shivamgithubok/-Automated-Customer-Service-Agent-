import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
import PyPDF2
import chardet
from typing import List, Tuple, Dict
import hashlib
import uuid


# Load environment variables
env_path = Path(__file__).parent.parent / '.env'

if env_path.exists():
    load_dotenv(dotenv_path=env_path)


class RAGAgent:
    def __init__(self):
        # Setup paths
        self.base_path = Path(__file__).parent.parent
        self.faiss_path = self.base_path / "vectorstore"
        self.faiss_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize conversation history
        self.conversation_history: List[Tuple[str, str]] = []

        # Load API key and configure Google AI
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        try:
            # Use a local embedding model
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

            self.vector_store = None
            self.document_ids = {} 
            self.load_or_create_vectorstore()

        except Exception as e:
            raise

    def load_or_create_vectorstore(self):
        """Load existing FAISS index or create a new one"""
        try:
            if self.faiss_path.exists() and os.listdir(self.faiss_path):
                # --- ADD THIS ---
                print(f"RAGAgent: Found existing vector store at {self.faiss_path}. Loading...")
                self.vector_store = FAISS.load_local(
                    folder_path=str(self.faiss_path),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                # --- ADD THIS ---
                print(f"RAGAgent: Vector store loaded successfully with {self.vector_store.index.ntotal} vectors.")
            else:
                # --- ADD THIS ---
                print("RAGAgent: No vector store found. Creating a new one with initial setup text.")
                self.vector_store = FAISS.from_texts(
                    texts=["Initial setup"],
                    embedding=self.embeddings,
                    metadatas=[{"source": "initial"}]
                )
                self.save_vectorstore()
                print("RAGAgent: New vector store created and saved.")
        except Exception as e:
            # --- ADD THIS ---
            print(f"RAGAgent: CRITICAL - Failed to load or create vector store: {e}")
            raise

    def save_vectorstore(self):
        """Save the FAISS index to disk"""
        try:
            if self.vector_store is None:
                raise ValueError("No vector store to save")

            self.faiss_path.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(self.faiss_path))
        except Exception as e:
            raise ValueError(f"Error saving vector store: {str(e)}")

    def process_document(self, file_path: str) -> dict:
        """Process and store document embeddings with unique IDs, avoiding duplicates."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            file_extension = file_path.suffix.lower()

            if file_extension == '.pdf':
                document_text = self.extract_text_from_pdf(str(file_path))
            elif file_extension == '.txt':
                document_text = self.read_text_file(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_extension}. Only .pdf and .txt files are supported.")

            document_hash = self.generate_document_hash(document_text)

            if document_hash in self.document_ids:
                return {
                    "status": "success",
                    "message": "Document already processed",
                    "document_id": self.document_ids[document_hash]
                }

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            texts = text_splitter.split_text(document_text)

            # Create unique IDs for each document chunk
            docs = []
            for i, text_chunk in enumerate(texts):
                chunk_id = f"{file_path.name}-{i}"
                docs.append(Document(page_content=text_chunk, metadata={"source": file_path.name, "chunk_id": chunk_id}))

            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(docs, self.embeddings)
            else:
                self.vector_store.add_documents(docs)

            self.document_ids[document_hash] = file_path.name
            self.save_vectorstore()
            
            print(f"Document {file_path.name} processed and added with ID {self.document_ids[document_hash]}")
            return {
                "status": "success",
                "chunks_processed": len(texts),
                "file_name": Path(file_path).name,
                "file_type": file_extension,
                "document_id": self.document_ids[document_hash]
            }

        except Exception as e:
            raise ValueError(f"Error processing document: {str(e)}")

    def generate_document_hash(self, document_text: str) -> str:
        """Generate a hash value for the document to identify it uniquely."""
        return hashlib.sha256(document_text.encode('utf-8')).hexdigest()

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
                return "\n".join(text)
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {str(e)}")

    def read_text_file(self, file_path: str) -> str:
        """Read text from a file with proper encoding detection"""
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] if result['encoding'] else 'utf-8'
            
            return raw_data.decode(encoding)
        except Exception as e:
            raise ValueError(f"Error reading text file: {str(e)}")

    def query(self, question: str, k: int = 4) -> dict:
        """Query the vector store and generate a response with context and justification."""
        print("\n--- RAG AGENT QUERY ---")
        print(f"Received question: '{question}'")
        try:
            if not self.vector_store:
                return {"error": "No documents have been processed yet"}
            print(f"Performing similarity search. Index has {self.vector_store.index.ntotal} total vectors.")
            
            relevant_docs = self.vector_store.similarity_search(question, k=k)
            
            if not relevant_docs:
                return {"answer": "No relevant documents found.", "justification": "", "context": "", "source_documents": []}

            context = "\n".join([doc.page_content for doc in relevant_docs])

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.7,
            )
            print(f"Context length: {len(context)} characters")
            # This is the prompt template that includes logic for both Resumes and Legal Documents
            prompt_template = PromptTemplate(
                input_variables=["question", "context"],
                template="""
                You are a helpful assistant. First, analyze the context below to determine if it is from a Resume or a Legal Document.
                Then, answer the question based on the document type and the provided context.
                
                Context: {context}
                -------------------------------------------------------
                Question: {question}
                -------------------------------------------------------
                
                **Instructions:**

                **1. If the document is a RESUME:**
                - Provide a direct and concise answer to the question using only the information in the context.
                - Use the following examples to guide your response format.
                
                -------------------------------------------------------
                    ## Example Question:
                    "What is the education background of the individual?"

                    ## Answer (provide only the specific information requested):
                    B.Tech in Electronics and Communication Engineering (Specialization in VLSI) Nov 2022 - Present, CGPA: 7.76

                    -------------------------------------------------------
                    ## Example Question:
                    "What are the technical skills of the person?"

                    ## Answer (provide only the specific information requested):
                    Machine Learning, Deep Learning, Image Processing, NLP, Transformers, GenerativeAI, LLM, langchain, langgraph, Agent, etc.

                    -------------------------------------------------------
                    ## Example Question:
                    "Can you tell me about the professional experience?"

                    ## Answer (provide only the specific information requested):
                    AI Intern at Edunet Foundation, built plant disease detection model with CNNs, reached 95% accuracy.

                    -------------------------------------------------------
                    ## Example Question:
                    "what is his github account"

                    ## Answer (provide only the specific information requested):
                    https://github.com/username
                    
                    --------------------------------------------------------
                    ## Example Question:
                    "What is the person's LinkedIn profile?"

                    ## Answer (provide only the specific information requested):
                    https://www.linkedin.com/in/username
                    
                    ---------------------------------------------------------

                    ## Example Question:
                    "What is the person's Kaggle profile"

                    ## Answer (provide only the specific information requested):
                    https://www.kaggle.com/username

                    ----------------------------------------------------------

                    ## Example Question:
                    "What is the email in the documents"

                    ## Answer (provide only the specific information requested):
                    example@email.com, example2@email.ac.in , Email: shivachadhhary@gmail.com 

                    ----------------------------------------------------------

                    ## Example Question:
                    " Position of Responsibility"

                    ## Answer (provide only the specific information requested):
                    VLSI Club Associate Member, General Secretary at Mekanika (Sep 2023- Present), NSS Unit Leader (Dec 2022- Apr 2024)
                    
                    ----------------------------------------------------------

                    ## Example Question:
                    "Describe name project"

                    ## Answer (provide only the specific information requested):
                    .Implemented dialogue summarization by adapting BART-Large-CNN to the SAMSum dataset (16,369 dialogues),
                    .Optimized AutoTokenizer, reducing processing time by 30%.
                    .Secured a ROUGE-2 score of 0.23 maintaining crucial dialogue context.                        
                    -----------------------------------------------------------

                    ## Example Question:
                    "Achievements"

                    ## Answer (provide only the specific information requested):
                    VLSI Club Associate Member, Kaggle Contributor, Top 30 in CodeRush by Codeforces Master

                    ----------------------------------------------------------

                    ## Example Question:
                    "What projects has the person worked on?"

                    ## Answer (provide only the specific information requested):
                    Dialogue Summarization Using BART-Large-CNN, Smart-Assistant-for-Research-Summarization, Spam Message Detection, etc.

                    -------------------------------------------------------
                    ## Example Question:
                    "What extracurricular activities has the person been involved in?"

                    ## Answer (provide only the specific information requested):
                    VLSI Club Associate Member, Kaggle Contributor, Top 30 in CodeRush by Codeforces Master

                **2. If the document is a LEGAL DOCUMENT and the question asks for a summary or analysis:**
                - You must act as a specialized AI legal assistant for a top-tier US law firm.
                - Your entire output must be a single, minified JSON object. Do not include any text, explanations, or markdown formatting before or after the JSON.
                - Based *only* on the provided context, perform the analysis and structure your response with the keys "summary", "key_points", and "risks".

                - ### JSON Structure for Legal Document Analysis:
                  {{
                    "summary": "A single, dense paragraph providing a high-level overview of the document's purpose, key parties, and primary legal implications.",
                    "key_points": [
                      "Critical article/clause 1: Brief, one-sentence explanation of its direct significance.",
                      "Critical article/clause 2: Brief, one-sentence explanation of its direct significance."
                    ],
                    "risks": [
                      "Proactively flagged ambiguous language or potential liability.",
                      "Identification of any elements that could lead to future disputes."
                    ]
                  }}

                **3. If the question is about a specific detail in a Legal Document (e.g., "Who are the parties involved?"):**
                - Provide a direct answer to the question without using the JSON format.
                
                Please provide your answer now.
                """
            )
            
            prompt = prompt_template.format(question=question, context=context)
            response = llm.invoke(prompt)
            answer = response.content.strip()

            justification = f"Answer based on {len(relevant_docs)} relevant documents"
            self.conversation_history.append((question, answer))

            return {
                "answer": answer or "No answer found",
                "justification": justification,
                "context": context,
                "source_documents": [doc.metadata for doc in relevant_docs],
                "num_docs_found": len(relevant_docs)
            }
        except Exception as e:
            return {"error": f"Error processing query: {str(e)}"}