from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.api import router as api_router
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Document Analysis API",
    description="Upload documents and ask questions.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API router
# All routes from api.py will be available under the /api prefix
# e.g., /api/, /api/upload, /api/ask
app.include_router(api_router, prefix="/api")

@app.get("/")
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "API is running. Go to /api for the main routes."}

# This block allows running the script directly for development
if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)