import json
import os
import base64
import time
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, Body, HTTPException
from mangum import Mangum
from pydantic import BaseModel

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finance-rag-api")

# Create FastAPI app
app = FastAPI(title="Finance RAG API")

# Import app modules conditionally to reduce cold start time
def _import_rag():
    # Only import when needed to reduce cold start
    from src.rag.finance_rag import FinanceRAG
    return FinanceRAG()

def _import_db():
    # Only import when needed
    from src.db.database import Database
    return Database()

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
        logger.info("Initializing RAG instance")
        _rag_instance = _import_rag()
    return _rag_instance

# Request size limit (to stay within free tier constraints)
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
        response = rag.process_query(request.query)

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

        # Process upload
        db = _import_db()
        result = db.import_transactions_from_csv_content(file_content)

        return {
            "message": "Upload successful",
            "transactions_imported": result.get("imported", 0)
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Create Lambda handler
handler = Mangum(app)

# Legacy handlers for backward compatibility
def query_handler(event, context):
    return handler(event, context)

def upload_handler(event, context):
    return handler(event, context)

def feedback_handler(event, context):
    return handler(event, context)

def options_handler(event, context):
    """Handle OPTIONS requests for CORS preflight"""
    return _build_response(200, "") 