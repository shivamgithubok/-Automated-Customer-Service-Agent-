from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
import os
import shutil
from pathlib import Path
import requests
from config import configuration
from agents.Rag_agent import RAGAgent
from agents.scraping_agents import Scraper
from agents.Mcp_agents import mcp_ask
import logging
from agents.HTML_page_agent import scrape_website, answer_query

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

# class AskScrapRequest(BaseModel):
#     question: str

#     class Config:
#         schema_extra = {
#             "example": {
#                 "question": "What information can you find about this topic?"
#             }
#         }

# class ScrapRequest(BaseModel):
#     url: str

#     class Config:
#         schema_extra = {
#             "example": {
#                 "url": "https://en.wikipedia.org/wiki/Example"
#             }
#         }

# class McpRequest(BaseModel):
#     question: str
    
# class ScrapRequest(BaseModel):
#     url: str

#     class Config:
#         schema_extra = {
#             "example": {
#                 "url": "https://example.com"
#             }
#         }

#     class Config:
#         schema_extra = {
#             "example": {
#                 "question": "What is quantum computing?"
#             }
#         }


# class ScrapRequest(BaseModel):
#     url: str


# class AskScrapRequest(BaseModel):
#     url: str
#     question: str


@app.get("/")
def home():
    return {"message": "Welcome to the backend service!"}

# @app.post("/ask-mcp")
# async def ask_mcp(request: McpRequest):
#     try:
#         # Call MCP agent with the question
#         answer = mcp_ask(request.question)
        
#         return {
#             "answer": answer,
#             "source": "MCP Agent"
#         }
#     except Exception as e:
#         logger.error(f"Error in MCP processing: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing MCP query: {str(e)}"
#         )

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
        print(f"Received request: {request}")
        if not rag_agent:
            raise HTTPException(
                status_code=500,
                detail="RAG Agent not initialized. Please check server logs."
            )
        # Process the query
        query_result = rag_agent.query(request.question)
        print(f"Received question: {request.question}")

        # Structure the response to include both question and answer
        response = {
            "question": request.question,
            "answer": query_result.get("answer"),
            "source_documents": query_result.get("source_documents", []),
            "num_docs_found": query_result.get("num_docs_found", 0)
        }
        print(f"Response: {response}")
        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )




# @app.post("/scrap")
# async def scrap_website_endpoint(req: ScrapRequest):
#     try:
#         if not req.url:
#             raise HTTPException(status_code=400, detail="URL is required")
            
#         if not req.url.startswith(('http://', 'https://')):
#             raise HTTPException(status_code=400, detail="Invalid URL format. URL must start with http:// or https://")
            
#         scrap_agent = Scraper()
#         scraped_data = scrap_agent.scrape_website(req.url)
        
#         if not scraped_data:
#             raise HTTPException(status_code=500, detail="Failed to scrape website")
            
#         return {
#             "status": "success",
#             "message": "Website scraped successfully",
#             "data": scraped_data
#         }
        
#     except HTTPException as he:
#         raise he
#     except requests.RequestException as e:
#         raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
#     except Exception as e:
#         logger.error(f"Scraping error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error scraping website: {str(e)}")



# @app.post("/scrap")
# def scrap_endpoint(req: ScrapRequest):
#     """Endpoint to scrape and store website content"""
#     scrap_agents = Scraper()
#     result = scrap_agents.scrape_website(req.url)
#     return {
#         "status": "success",
#         "message": "Website content scraped and stored successfully",
#         "data": result
#     }


# @app.post("/ask_scrap")
# def ask_scrap_endpoint(req: AskScrapRequest):
#     """Endpoint to ask questions about previously scraped content"""
#     scrap_agents = Scraper()
#     try:
#         answer = scrap_agents.ask_question(req.question)
#         return {
#             "status": "success",
#             "message": "Question answered successfully",
#             "data": answer
#         }
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error answering question: {str(e)}"
#         )
    

# # Storage for scraped content
# # scraped_data = {}

# # Pydantic model for input data
# class ScrapeRequest(BaseModel):
#     url: str

# class QueryRequest(BaseModel):
#     query: str


# @app.post("/scrape")
# async def scrape(scrape_request: ScrapeRequest):
#     result = scrape_website(scrape_request.url)
#     return result


# # FastAPI route to query scraped content by class or id
# @app.get("/query")
# async def query(query_request: QueryRequest):
#     result = answer_query(query_request.query)
#     return result



if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)