from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
from pydantic import BaseModel
import os
import shutil
from pathlib import Path
import logging
import traceback
import uuid
from agents.Rag_agent  import RAGAgent

# Setup logger, router, etc.
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
router = APIRouter()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- INITIALIZE THE NEW AGENT ---
try:
    logger.info("Initializing RAGAgent...")
    agent = RAGAgent()
    logger.info("RAGAgent initialized successfully.")
except Exception as e:
    logger.error(f"FATAL: Failed to initialize Agent: {e}")
    logger.error(traceback.format_exc())
    agent = None

# --- CONVERSATION MEMORY (In-memory implementation) ---
# In a real production app, use Redis, a database, or a session store.
conversations = {}

# --- Pydantic Models ---
class AskRequest(BaseModel):
    question: str
    session_id: str = None # Optional: to maintain conversation history

# --- API Endpoints ---
@router.get("/status")
def get_status():
    if not agent:
        raise HTTPException(status_code=500, detail="Agent failed to initialize.")
    return {"status": "ok", "agent_initialized": True}

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized.")
    
    file_paths = []
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(str(file_path))

    try:
        # The new agent can process multiple files at once
        agent.process_documents(file_paths)
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing files: {e}")
    finally:
        # Clean up uploaded files
        for path in file_paths:
            os.remove(path)

    return {"message": f"Successfully processed {len(file_paths)} files."}

@router.post("/ask")
async def ask(request: AskRequest):
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized.")

    # --- Part 3: Manage Conversational Memory ---
    session_id = request.session_id or str(uuid.uuid4())
    
    # Retrieve chat history or create a new one
    chat_history = conversations.get(session_id, [])

    try:
        # Get the answer from the new agent, providing history
        answer = agent.query(request.question, chat_history)

        # Update the history with the new question and answer
        chat_history.append({"role": "user", "content": request.question})
        chat_history.append({"role": "assistant", "content": answer})
        conversations[session_id] = chat_history

        return {
            "answer": answer,
            "session_id": session_id,
            "chat_history": chat_history,
        }
    except Exception as e:
        logger.error(f"Error during query for session {session_id}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="An internal agent error occurred.")