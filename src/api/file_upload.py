from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Depends
import os
import uuid
import tempfile
import time
import logging
from typing import Dict, List, Optional, Any
import shutil
import asyncio
import math
import json

# Import our data processing modules
from data_processing.processor import DataProcessor
from data_processing.process_data import process_file_to_db
from db.database import Database

# Configure logging
logger = logging.getLogger("finance-llm-upload")

# Create router
router = APIRouter()

# Track upload progress
upload_tasks = {}

async def process_file_task(file_path: str, original_filename: str, task_id: str, db: Database):
    """Background task to process uploaded file"""
    try:
        logger.info(f"Starting processing of file {original_filename} (task {task_id})")

        # Update task status
        upload_tasks[task_id]["status"] = "processing"

        # Ensure DB connection is valid (may have been created in a different thread)
        try:
            # Test connection with a simple query
            if db.conn is None or not hasattr(db.conn, 'cursor'):
                logger.info("Reconnecting to database in background task")
                db._connect()

            # Ensure tables exist
            db.create_tables()
        except Exception as db_err:
            logger.warning(f"Database connection issue in background task: {db_err}. Will create new connection.")
            # Create a new database connection for this thread
            db = Database()
            db.create_tables()

        # Process the file
        # Create input and output directories for DataProcessor
        input_dir = os.path.dirname(file_path)
        output_dir = os.path.join(tempfile.gettempdir(), f"finance_output_{task_id}")
        os.makedirs(output_dir, exist_ok=True)

        # Initialize DataProcessor with required parameters
        data_processor = DataProcessor(input_dir=input_dir, output_dir=output_dir)

        # Update task progress
        upload_tasks[task_id]["progress"] = 25
        upload_tasks[task_id]["status_message"] = "Detecting file format"

        # Process the file
        file_type = data_processor.detect_file_type(file_path)

        upload_tasks[task_id]["progress"] = 50
        upload_tasks[task_id]["status_message"] = f"Processing {file_type} file format"

        # Process the file to the database
        result = process_file_to_db(file_path, db)

        upload_tasks[task_id]["progress"] = 90
        upload_tasks[task_id]["status_message"] = "Finalizing import"

        # Simulate some processing time
        await asyncio.sleep(1)

        # Sanitize results to ensure JSON serializability
        sanitized_categories = []
        if isinstance(result.get("categories_detected"), (list, set, tuple)):
            # Filter out NaN values and convert to strings
            sanitized_categories = [
                str(cat) for cat in result.get("categories_detected", []) 
                if cat is not None and (not isinstance(cat, float) or not math.isnan(cat))
            ]

        # Ensure date range is a string
        date_range = str(result.get("date_range", "")) if result.get("date_range") is not None else ""

        # Update task status to complete
        upload_tasks[task_id]["status"] = "completed"
        upload_tasks[task_id]["progress"] = 100
        upload_tasks[task_id]["status_message"] = "Import completed successfully"
        upload_tasks[task_id]["result"] = {
            "transactions_processed": int(result.get("transactions_processed", 0)),
            "categories_detected": sanitized_categories,
            "date_range": date_range
        }

        logger.info(f"File {original_filename} processed successfully (task {task_id})")

    except Exception as e:
        logger.error(f"Error processing file {original_filename} (task {task_id}): {str(e)}", exc_info=True)
        upload_tasks[task_id]["status"] = "failed"
        upload_tasks[task_id]["status_message"] = f"Error: {str(e)}"
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error removing temporary file {file_path}: {str(e)}")

@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Database = Depends(lambda: Database())
):
    """Endpoint to upload financial data files"""
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate file extension
    allowed_extensions = [".csv", ".ofx", ".qfx", ".json", ".xlsx"]
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported formats: {', '.join(allowed_extensions)}"
        )

    # Create a temporary file
    task_id = str(uuid.uuid4())
    temp_file_path = ""

    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        # Initialize task tracking
        upload_tasks[task_id] = {
            "id": task_id,
            "filename": file.filename,
            "upload_time": time.time(),
            "status": "queued",
            "progress": 0,
            "status_message": "Upload received, waiting for processing"
        }

        # Start background processing task
        background_tasks.add_task(
            process_file_task,
            temp_file_path,
            file.filename,
            task_id,
            db
        )

        return {
            "task_id": task_id,
            "filename": file.filename,
            "status": "queued",
            "message": "File uploaded and queued for processing"
        }

    except Exception as e:
        # Clean up temp file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        logger.error(f"Error uploading file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")
    finally:
        file.file.close()

@router.get("/upload/status/{task_id}")
async def get_upload_status(task_id: str):
    """Get the status of a file upload task"""
    if task_id not in upload_tasks:
        raise HTTPException(status_code=404, detail="Upload task not found")

    # Ensure the task data is JSON serializable
    # Function to sanitize values
    def sanitize_for_json(obj):
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [sanitize_for_json(item) for item in obj]
        elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return str(obj)  # Convert NaN/inf to string
        elif obj is None:
            return None
        return obj

    # Sanitize the task data
    task_data = sanitize_for_json(upload_tasks[task_id])

    # Verify it's JSON serializable
    try:
        json.dumps(task_data)
    except Exception as e:
        logger.error(f"Error serializing task data: {e}")
        # Fallback to basic task status
        return {
            "id": task_id,
            "status": upload_tasks[task_id].get("status", "unknown"),
            "error": "Could not serialize complete task data"
        }

    return task_data

@router.get("/uploads")
async def list_uploads():
    """List all uploads and their status"""
    # Ensure all task data is JSON serializable
    # Function to sanitize values
    def sanitize_for_json(obj):
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [sanitize_for_json(item) for item in obj]
        elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return str(obj)  # Convert NaN/inf to string
        elif obj is None:
            return None
        return obj

    # Sanitize all tasks
    tasks = [sanitize_for_json(task) for task in upload_tasks.values()]

    return tasks 