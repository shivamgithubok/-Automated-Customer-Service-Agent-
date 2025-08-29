from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
import os
import shutil
from pathlib import Path
from config import configuration
from agents.Rag_agent import RAGAgent
from agents.scraping_agents import Scraper
import logging


# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Initialize RAG Agent
try:
    rag_agent = RAGAgent()
except Exception as e:
    rag_agent = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskAnythingRequest(BaseModel):
    question: str

    class Config:
        schema_extra = {
            "example": {
                "question": "What is the capital of France?"
            }
        }


class ScrapRequest(BaseModel):
    url: str


class AskScrapRequest(BaseModel):
    url: str
    question: str


@app.get("/")
def home():
    return {"message": "Welcome to the backend service!"}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        if not rag_agent:
            raise HTTPException(
                status_code=500,
                detail="RAG Agent not initialized. Please check server logs."
            )

        uploaded_files = []
        for file in files:
            logger.info(f"Processing file: {file.filename}")
            # Save file temporarily
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process the file with RAG agent
            try:
                upload_result = rag_agent.process_document(str(file_path))
                uploaded_files.append({
                    "filename": file.filename,
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                uploaded_files.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })
            
            # Clean up the temporary file
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove file {file_path}: {str(e)}")

        return {
            "message": "Files processed",
            "details": uploaded_files
        }

    except Exception as e:
        logger.error(f"Error handling file upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error handling file upload: {str(e)}"
        )


@app.post("/ask", response_model=dict) 
async def ask(request: AskAnythingRequest):
    try:
        if not rag_agent:
            raise HTTPException(
                status_code=500,
                detail="RAG Agent not initialized. Please check server logs."
            )

        # Process the query
        query_result = rag_agent.query(request.question)
        
        # Structure the response to include both question and answer
        response = {
            "question": request.question,
            "answer": query_result.get("answer"),
            "source_documents": query_result.get("source_documents", []),
            "num_docs_found": query_result.get("num_docs_found", 0)
        }
        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )




@app.post("/scrap")
def scrap_website_endpoint(req: ScrapRequest):
    try:
        scrap_agents = Scraper()
        scraped_data = scrap_agents.scrape_website(req.url)
        # print(scraped_data[:100])
        return {"message": "Website scraped successfully", "data": scraped_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping website: {str(e)}")



@app.post("/ask_scrap")
def ask_scrap_endpoint(req: AskScrapRequest):
    scrap_agents = Scraper()
    scraped_data = scrap_agents.scrape_website(req.url)
    answer = scrap_agents.ask_question(scraped_data, req.question)
    return {"message": "Question answered successfully", "data": answer}


# Run the application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)