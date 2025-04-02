import os
import sys
import json
import base64
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finance-rag-api")

# Create FastAPI app
app = FastAPI(title="Finance RAG API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class QueryRequest(BaseModel):
    query: str

class UploadRequest(BaseModel):
    file: str
    fileName: str = "upload.csv"
    isBase64Encoded: bool = False

class FeedbackRequest(BaseModel):
    query_id: str
    rating: int
    comments: str = ""

# Cache for FinanceRAG instance to improve performance across invocations
_rag_instance = None

def _get_rag_instance():
    """Get or create the FinanceRAG instance with connection pooling"""
    global _rag_instance
    if _rag_instance is None:
        # Always use the real implementation
        from src.rag.finance_rag import FinanceRAG
        logger.info("Initializing Hugging Face RAG instance")
        _rag_instance = FinanceRAG(
            llm_provider="huggingface",
            llm_model="mistralai/Mistral-7B-Instruct-v0.2"
        )
    return _rag_instance

# Request size limit
MAX_REQUEST_SIZE_BYTES = 5 * 1024 * 1024  # 5MB

# FastAPI routes
@app.post("/query")
async def query(request: QueryRequest):
    """Handle financial query requests"""
    logger.info("Processing query request")
    
    try:
        # Initialize RAG
        start_time = time.time()
        rag = _get_rag_instance()
        
        # Process query
        response = rag.query(request.query)
        
        # Measure and log performance
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Query processed in {processing_time:.2f}ms")
        
        # Include performance data in response
        return {
            "response": response,
            "performance": {
                "processing_time_ms": processing_time
            }
        }
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/upload")
async def upload(request: UploadRequest):
    """Handle financial data uploads"""
    logger.info("Processing upload request")
    
    try:
        # Process file content
        if request.isBase64Encoded:
            file_content = base64.b64decode(request.file).decode('utf-8')
        else:
            file_content = request.file
            
        if not file_content:
            raise HTTPException(status_code=400, detail="Missing file content")
        
        # Check file size
        if len(file_content.encode('utf-8')) > MAX_REQUEST_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Always use the real database implementation
        from src.db.database import Database
        db = Database()
        result = db.import_transactions_from_csv_content(file_content)
        
        # Automatically sync database with vector store
        rag = _get_rag_instance()
        sync_result = rag.synchronize_db_and_vector_store()
        
        return {
            "message": "Upload successful",
            "transactions_imported": result.get("imported", 0),
            "sync_status": "success" if sync_result.get("success", False) else "error"
        }
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    """Handle user feedback on responses"""
    logger.info("Processing feedback request")
    
    try:
        # Store feedback (simplified implementation)
        # In production, this would store to a database
        logger.info(f"Received feedback for query {request.query_id}: rating={request.rating}, comments={request.comments}")
        
        return {"message": "Feedback recorded"}
    
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/sync")
async def sync_db_and_vector_store():
    """Synchronize database and vector store"""
    logger.info("Processing sync request")
    
    try:
        # Initialize RAG
        start_time = time.time()
        rag = _get_rag_instance()
        
        # Synchronize DB and vector store
        result = rag.synchronize_db_and_vector_store()
        
        # Measure and log performance
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Sync processed in {processing_time:.2f}ms")
        
        # Include performance data in response
        return {
            "status": "success" if result.get("success", False) else "error",
            "result": result,
            "performance": {
                "processing_time_ms": processing_time
            }
        }
    
    except Exception as e:
        logger.error(f"Error processing sync: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "environment": os.getenv("STAGE", "development")}

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Finance RAG API",
        "version": "1.0.0",
        "description": "API for financial RAG system",
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "Health check endpoint"},
            {"path": "/query", "method": "POST", "description": "Process financial queries"},
            {"path": "/upload", "method": "POST", "description": "Upload financial data"},
            {"path": "/feedback", "method": "POST", "description": "Submit feedback on responses"},
            {"path": "/sync", "method": "POST", "description": "Synchronize database and vector store"}
        ]
    }

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 