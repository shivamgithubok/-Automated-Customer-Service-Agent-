from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import os
import shutil
from pathlib import Path
import logging
import traceback
import uuid
from agents.Rag_agent import RAGAgent
from agents.Mcp_agents import run_mcp_agent, load_documents_into_vectorstore
from agents.scraping_agents import ScrapingAgent

# --- Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
router = APIRouter()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Initialize BOTH agents ---
try:
    logger.info("Initializing RAG Agent for document-specific Q&A...")
    rag_agent = RAGAgent()
    logger.info("RAG Agent initialized.")

    # The MCP agent will automatically load from the same vector store
    logger.info("MCP Agent for general reasoning is also ready.")

except Exception as e:
    logger.error(f"FATAL: Failed to initialize agents: {e}")
    logger.error(traceback.format_exc())
    rag_agent = None

# --- In-memory conversation store for the RAG agent ---
conversations_rag = {}

# --- Pydantic Models for different request types ---
class AskRagRequest(BaseModel):
    question: str
    session_id: str = None

class AskMcpRequest(BaseModel):
    question: str

# --- API Endpoints ---

@router.get("/status")
def get_status():
    if not rag_agent:
        raise HTTPException(status_code=500, detail="Agents failed to initialize.")
    return {"status": "ok", "message": "All agents are running."}

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Uploads documents to the SHARED knowledge base for BOTH agents.
    """
    if not rag_agent:
        raise HTTPException(status_code=500, detail="Agents not initialized.")
    
    file_paths = []
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(str(file_path))

    try:
        # Use the RAG agent's powerful document processor
        rag_agent.process_documents(file_paths)
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing files: {e}")
    finally:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)

    return {"message": f"Successfully processed and added {len(file_paths)} files to the shared knowledge base."}

@router.post("/ask-rag")
async def ask_rag_agent(request: AskRagRequest):
    """
    Ask a question specifically to the RAG Agent.
    Use this for in-depth, conversational queries about uploaded documents.
    """
    if not rag_agent:
        raise HTTPException(status_code=500, detail="RAG Agent not initialized.")

    # Validate request
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    session_id = request.session_id or str(uuid.uuid4())
    chat_history = conversations_rag.get(session_id, [])

    try:
        # Get response from RAG agent
        response = rag_agent.query(request.question, chat_history)
        
        if isinstance(response, dict) and "error" in response:
            raise HTTPException(status_code=500, detail=response["error"])
        
        # Update chat history
        chat_history.append({"role": "user", "content": request.question})
        chat_history.append({"role": "assistant", "content": response["answer"] if isinstance(response, dict) else response})
        conversations_rag[session_id] = chat_history[-10:]  # Keep last 10 messages

        return {
            "agent_used": "RAG Agent",
            "answer": response["answer"] if isinstance(response, dict) else response,
            "context": response.get("context") if isinstance(response, dict) else None,
            "sources": response.get("source_documents") if isinstance(response, dict) else None,
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"Error in RAG Agent query: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred with the RAG agent: {str(e)}"
        )

@router.post("/ask-mcp")
async def ask_mcp_agent(request: AskMcpRequest):
    """
    Ask a question to the MCP Reasoning Agent.
    Use this for any general or critical question. It will decide whether to use
    your documents, the web, or other tools.
    """
    try:
        logger.info(f"Received question for MCP Agent: {request.question}")
        answer = run_mcp_agent(request.question)
        return {
            "agent_used": "MCP Agent",
            "answer": answer
        }
    except Exception as e:
        logger.error(f"An error occurred in the MCP agent: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred with the MCP agent."
        )
    

try:
    logger.info("Initializing Scraping Agent...")
    scraping_agent = ScrapingAgent()
    logger.info("Scraping Agent initialized.")
except Exception as e:
    logger.error(f"FATAL: Failed to initialize Scraping Agent: {e}")
    scraping_agent = None



# --- CORRECTED Pydantic Models for Scraping ---
class ScrapeRequest(BaseModel):
    """Model for the scrape request."""
    url: str
    # Add a flag to let the user choose the scraping method.
    # Defaults to False for faster, simpler scraping.
    dynamic: bool = False

class AskScrapedRequest(BaseModel):
    """Model for asking a question to a scraped URL."""
    url: str
    question: str

# --- Corrected Endpoints ---

@router.post("/scrape")
async def scrape_url(request: ScrapeRequest):
    """
    Scrapes a single URL, caching its HTML and vectorizing its text content.
    
    - **url**: The URL to scrape.
    - **dynamic**: Set to `true` for websites that rely on JavaScript to render content 
      (e.g., React, Angular, Vue sites). Defaults to `false`.
    """
    # This check is good, keep it.
    if not scraping_agent:
        raise HTTPException(status_code=500, detail="Scraping Agent not initialized.")
    
    try:
        # --- KEY CORRECTION ---
        # Pass the 'dynamic' flag from the request to the agent's method.
        logger.info(f"Scraping URL: {request.url} with dynamic mode: {request.dynamic}")
        result = scraping_agent.scrape_website(url=request.url, dynamic=request.dynamic)
        
        return {"message": "Scraping successful", "details": result}
    
    except ValueError as ve:
        # Catches specific errors from the agent for better feedback
        logger.error(f"Value error during scraping {request.url}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
        
    except Exception as e:
        # General catch-all for other errors (e.g., network issues, timeouts)
        logger.error(f"An unexpected error occurred during scraping {request.url}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post("/ask-scraped")
async def ask_scraped_url(request: AskScrapedRequest):
    """
    Asks a question about a previously scraped URL. Can handle general questions
    or specific requests to extract data from classes/IDs into a table.
    """
    if not scraping_agent:
        raise HTTPException(status_code=500, detail="Scraping Agent not initialized.")
    
    try:
        logger.info(f"Asking question about URL: {request.url}")
        answer = scraping_agent.ask(question=request.question, url=request.url)
        return {"answer": answer}
        
    except ValueError as ve:
        # This could happen if the URL was never scraped first
        logger.error(f"Value error while asking {request.url}: {ve}")
        raise HTTPException(status_code=404, detail=str(ve)) # 404 is more appropriate here
        
    except Exception as e:
        logger.error(f"Error during ask-scraped query for {request.url}: {e}")
        raise HTTPException(status_code=500, detail=str(e))