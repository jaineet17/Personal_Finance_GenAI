from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional, Any
import logging
import time
import uuid
from pydantic import BaseModel, Field

# Import database module
from db.database import Database

# Configure logging
logger = logging.getLogger("finance-llm-feedback")

# Create router
router = APIRouter()

# Feedback schema
class FeedbackModel(BaseModel):
    query_id: str = Field(..., description="ID of the query this feedback is for")
    rating: int = Field(..., description="Rating from 1-5", ge=1, le=5)
    comment: Optional[str] = Field(None, description="Optional comment")
    category: Optional[str] = Field(None, description="Feedback category (accuracy, relevance, etc.)")
    original_query: Optional[str] = Field(None, description="Original query text")
    response_text: Optional[str] = Field(None, description="Response text that was rated")
    suggestion: Optional[str] = Field(None, description="Suggestion for improvement")

# Store feedback in memory (would be replaced with database in production)
feedback_store = {}

# Create feedback table if it doesn't exist
def init_feedback_table(db: Database):
    """Initialize the feedback table in the database if it doesn't exist"""
    try:
        db.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            query_id TEXT NOT NULL,
            rating INTEGER NOT NULL,
            comment TEXT,
            category TEXT,
            original_query TEXT,
            response_text TEXT,
            suggestion TEXT,
            timestamp REAL NOT NULL
        )
        """)
    except Exception as e:
        logger.error(f"Error creating feedback table: {str(e)}")
        raise

@router.post("/feedback")
async def submit_feedback(
    feedback: FeedbackModel,
    db: Database = Depends(lambda: Database())
):
    """Submit feedback for a query response"""
    # Generate ID for this feedback
    feedback_id = str(uuid.uuid4())
    timestamp = time.time()
    
    try:
        # Initialize feedback table if it doesn't exist
        init_feedback_table(db)
        
        # Store feedback in database
        db.execute(
            """
            INSERT INTO feedback 
            (id, query_id, rating, comment, category, original_query, response_text, suggestion, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, 
            (
                feedback_id,
                feedback.query_id,
                feedback.rating,
                feedback.comment,
                feedback.category,
                feedback.original_query,
                feedback.response_text,
                feedback.suggestion,
                timestamp
            )
        )
        
        # Also store in memory for quick access
        feedback_store[feedback_id] = {
            "id": feedback_id,
            "query_id": feedback.query_id,
            "rating": feedback.rating,
            "comment": feedback.comment,
            "category": feedback.category,
            "original_query": feedback.original_query,
            "response_text": feedback.response_text,
            "suggestion": feedback.suggestion,
            "timestamp": timestamp
        }
        
        return {
            "feedback_id": feedback_id,
            "status": "success",
            "message": "Feedback submitted successfully",
            "timestamp": timestamp
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

@router.get("/feedback/{feedback_id}")
async def get_feedback(
    feedback_id: str,
    db: Database = Depends(lambda: Database())
):
    """Get a specific feedback by ID"""
    # Check memory cache first
    if feedback_id in feedback_store:
        return feedback_store[feedback_id]
    
    # Otherwise, query database
    try:
        # Make sure table exists
        init_feedback_table(db)
        
        result = db.execute(
            """
            SELECT id, query_id, rating, comment, category, original_query, response_text, suggestion, timestamp 
            FROM feedback 
            WHERE id = ?
            """, 
            (feedback_id,)
        ).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="Feedback not found")
        
        # Convert to dict
        feedback_data = {
            "id": result[0],
            "query_id": result[1],
            "rating": result[2],
            "comment": result[3],
            "category": result[4],
            "original_query": result[5],
            "response_text": result[6],
            "suggestion": result[7],
            "timestamp": result[8]
        }
        
        # Update memory cache
        feedback_store[feedback_id] = feedback_data
        
        return feedback_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving feedback: {str(e)}")

@router.get("/feedback/query/{query_id}")
async def get_feedback_by_query(
    query_id: str,
    db: Database = Depends(lambda: Database())
):
    """Get all feedback for a specific query"""
    try:
        # Make sure table exists
        init_feedback_table(db)
        
        results = db.execute(
            """
            SELECT id, query_id, rating, comment, category, original_query, response_text, suggestion, timestamp 
            FROM feedback 
            WHERE query_id = ?
            ORDER BY timestamp DESC
            """, 
            (query_id,)
        ).fetchall()
        
        if not results:
            return []
        
        # Convert to list of dicts
        feedback_list = []
        for result in results:
            feedback_data = {
                "id": result[0],
                "query_id": result[1],
                "rating": result[2],
                "comment": result[3],
                "category": result[4],
                "original_query": result[5],
                "response_text": result[6],
                "suggestion": result[7],
                "timestamp": result[8]
            }
            feedback_list.append(feedback_data)
            
            # Update memory cache
            feedback_store[result[0]] = feedback_data
        
        return feedback_list
        
    except Exception as e:
        logger.error(f"Error retrieving feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving feedback: {str(e)}")

@router.get("/feedback")
async def list_feedback(
    limit: int = 100,
    offset: int = 0,
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
    db: Database = Depends(lambda: Database())
):
    """List all feedback with optional filtering"""
    try:
        # Make sure table exists
        init_feedback_table(db)
        
        # Build query
        query = """
            SELECT id, query_id, rating, comment, category, original_query, response_text, suggestion, timestamp 
            FROM feedback 
            WHERE 1=1
        """
        params = []
        
        # Add filters
        if min_rating is not None:
            query += " AND rating >= ?"
            params.append(min_rating)
        
        if max_rating is not None:
            query += " AND rating <= ?"
            params.append(max_rating)
        
        # Add order and limit
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        # Execute query
        results = db.execute(query, tuple(params)).fetchall()
        
        if not results:
            return []
        
        # Convert to list of dicts
        feedback_list = []
        for result in results:
            feedback_data = {
                "id": result[0],
                "query_id": result[1],
                "rating": result[2],
                "comment": result[3],
                "category": result[4],
                "original_query": result[5],
                "response_text": result[6],
                "suggestion": result[7],
                "timestamp": result[8]
            }
            feedback_list.append(feedback_data)
            
            # Update memory cache
            feedback_store[result[0]] = feedback_data
        
        return feedback_list
        
    except Exception as e:
        logger.error(f"Error listing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing feedback: {str(e)}")

@router.get("/feedback/stats")
async def get_feedback_stats(
    db: Database = Depends(lambda: Database())
):
    """Get statistics about feedback"""
    try:
        # Make sure table exists
        init_feedback_table(db)
        
        # Get count of feedback by rating
        rating_counts = db.execute(
            """
            SELECT rating, COUNT(*) 
            FROM feedback 
            GROUP BY rating 
            ORDER BY rating
            """
        ).fetchall()
        
        # Get average rating
        avg_rating = db.execute(
            """
            SELECT AVG(rating) 
            FROM feedback
            """
        ).fetchone()[0]
        
        # Get total count
        total_count = db.execute(
            """
            SELECT COUNT(*) 
            FROM feedback
            """
        ).fetchone()[0]
        
        # Format statistics
        rating_stats = {}
        for rating, count in rating_counts:
            rating_stats[str(rating)] = count
        
        return {
            "total_count": total_count,
            "average_rating": avg_rating,
            "rating_counts": rating_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting feedback stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting feedback stats: {str(e)}") 