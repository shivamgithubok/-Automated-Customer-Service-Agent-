import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
import PyPDF2
import chardet
from typing import List, Tuple, Dict
import hashlib



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
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )

            self.vector_store = None
            self.document_ids = {}  # Store document hashes here
            self.load_or_create_vectorstore()

        except Exception as e:
            raise

    def load_or_create_vectorstore(self):
        """Load existing FAISS index or create a new one"""
        try:
            if self.faiss_path.exists():
                self.vector_store = FAISS.load_local(
                    folder_path=str(self.faiss_path),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True  # We trust our own files
                )
            else:
                self.vector_store = FAISS.from_texts(
                    texts=["Initial setup"],
                    embedding=self.embeddings
                )
                self.save_vectorstore()
        except Exception as e:
            raise

    def save_vectorstore(self):
        """Save the FAISS index to disk"""
        try:
            if self.vector_store is None:
                raise ValueError("No vector store to save")

            # Ensure the directory exists
            self.faiss_path.mkdir(parents=True, exist_ok=True)

            # Save the FAISS index
            self.vector_store.save_local(str(self.faiss_path))
        except Exception as e:
            raise (f"Error saving vector store: {str(e)}")

    def process_document(self, file_path: str) -> dict:
        """Process and store document embeddings, avoiding duplicates."""
        try:
            # Convert to Path object and validate
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Simple file type detection based on extension
            file_extension = file_path.suffix.lower()

            if file_extension == '.pdf':
                document_text = self.extract_text_from_pdf(str(file_path))
            elif file_extension == '.txt':
                document_text = self.read_text_file(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_extension}. Only .pdf and .txt files are supported.")

            # Generate a unique hash ID for the document based on its content
            document_hash = self.generate_document_hash(document_text)

            # Check if the document ID already exists in the vector store
            if document_hash in self.document_ids:
                # Document already exists, return existing ID
                return {
                    "status": "success",
                    "message": "Document already processed",
                    "document_id": self.document_ids[document_hash]
                }

            # Split the document into smaller chunks for processing
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            texts = text_splitter.split_text(document_text)

            # Convert to documents
            docs = [Document(page_content=text) for text in texts]

            # Add documents to the vector store and store their ID
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(docs, self.embeddings)
            else:
                self.vector_store.add_documents(docs)

            # Store the document hash and associated ID for future checks
            self.document_ids[document_hash] = file_path.name  
            # Save updated vector store
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
                text = []
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
                return "\n".join(text)
        except Exception as e:
            raise (f"Error extracting text from PDF: {str(e)}")

    def read_text_file(self, file_path: str) -> str:
        """Read text from a file with proper encoding detection"""
        try:
            # Detect the encoding
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] if result['encoding'] else 'utf-8'

            # Read the file with detected encoding
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()

        except Exception as e:
            raise (f"Error reading text file: {str(e)}")

    def query(self, question: str, k: int = 2) -> dict:
        """Query the vector store and generate a response with context and justification."""
        try:
            if not self.vector_store:
                return {"error": "No documents have been processed yet"}

            # Search for relevant documents
            relevant_docs = self.vector_store.similarity_search(question, k=k)

            if not relevant_docs:
                return {"answer": "No relevant documents found.", "justification": "", "context": "", "source_documents": []}

            
            context = "\n".join([doc.page_content[:] + "..." for doc in relevant_docs])  # Limit to 150 chars per doc

            # Initialize LLM if not already done
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.7,
            )

            prompt_template = PromptTemplate(
                    input_variables=["question", "context"],
                    template="""
                        You are a helpful assistant. Analyze the question type and provide a focused response based on the context. The document contains several sections, each with a title. Please identify the sections, and answer based on the relevant section.

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

                        -------------------------------------------------------
                        Now, for the user's input, respond according to the question type:
                        ## Question:
                        {question}
                        -------------------------------------------------------
                        ## Context:
                        {context}

                        -------------------------------------------------------
                        ## Instructions:
                        1. Identify each **section** in the document (e.g., Education, Professional Experience, Projects, etc.).
                        2. Answer based on the content of the relevant section.
                        3. If the question relates to a section, use the content from that section to form the answer.
                    """
                )


            # Generate prompt
            prompt = prompt_template.format(
                question=question,
                context=context,
            )

            # Get response from LLM
            response = llm.invoke(prompt)
            answer = response.content.strip()

            # Create concise justification based on top document
            justification = f"This answer is based on the following context from the most relevant document: {relevant_docs[0].page_content[:300]}..."

            # Update conversation history
            self.conversation_history.append((question, answer))

            # Prepare response
            response = {
                "answer": answer or "No answer found",
                "justification": justification,
                "context": context,
                "source_documents": [doc.page_content[:300] for doc in relevant_docs[:1]],  
                "num_docs_found": len(relevant_docs)
            }
            return response

        except Exception as e:
            return {"error": f"Error processing query: {str(e)}"}