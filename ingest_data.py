#!/usr/bin/env python
"""
Unified Data Ingestion Tool for Finance LLM App

This script provides a single command to:
1. Process raw financial data from CSV files
2. Import the data into a single SQLite database
3. Sync the database with the vector store (ChromaDB)

Usage:
    python ingest_data.py [--clear-db] [--reset-vectors]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("data_ingestion")

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Ingest financial data into database and vector store")
    parser.add_argument('--clear-db', action='store_true', help='Clear existing database before importing')
    parser.add_argument('--reset-vectors', action='store_true', help='Reset vector store before syncing')
    args = parser.parse_args()
    
    logger.info("Starting unified data ingestion process")
    
    # Process raw data files first
    process_raw_data(clear_db=args.clear_db)
    
    # Sync with vector store
    sync_vector_store(reset_vectors=args.reset_vectors)
    
    logger.info("Data ingestion process completed")

def process_raw_data(clear_db=False):
    """Process raw data files and import into SQLite database"""
    logger.info("Processing raw data files...")
    
    # Import the process_data module
    try:
        # Add the src directory to the path if needed
        if not os.path.exists("src"):
            logger.warning("src directory not found in current path, using module import")
            from src.data_processing import process_data
        else:
            # Directly import if running from project root
            sys.path.insert(0, os.path.abspath("."))
            import src.data_processing.process_data as process_data
        
        # Use the non-interactive version with our params
        if clear_db:
            # Override the input function to automatically return 'y'
            original_input = __builtins__.input
            __builtins__.input = lambda _: 'y'
            try:
                process_data.main()
            finally:
                __builtins__.input = original_input
        else:
            # Override the input function to automatically return 'n'
            original_input = __builtins__.input
            __builtins__.input = lambda _: 'n'
            try:
                process_data.main()
            finally:
                __builtins__.input = original_input
        
        logger.info("Finished processing raw data files")
        
    except ImportError as e:
        logger.error(f"Error importing process_data module: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error processing raw data: {e}")
        sys.exit(1)

def sync_vector_store(reset_vectors=False):
    """Sync SQLite database with vector store"""
    logger.info("Syncing database with vector store...")
    
    try:
        # Import required modules
        from src.db.database import Database
        from src.embedding.vector_store import VectorStore
        from src.retrieval.retrieval_system import FinanceRetrieval
        
        # Initialize components
        db = Database()
        vector_store = VectorStore()
        retrieval = FinanceRetrieval(vector_store=vector_store)
        
        # Reset vector store if requested
        if reset_vectors:
            logger.info("Resetting vector store...")
            reset_result = vector_store.reset_vector_store()
            if reset_result.get("success", False):
                logger.info(f"Vector store reset completed successfully")
            else:
                logger.error(f"Vector store reset failed: {reset_result.get('error', 'Unknown error')}")
                
        # Sync with vector store
        logger.info("Syncing database with vector store...")
        sync_result = retrieval.sync_database_with_vector_store()
        
        logger.info(f"Added {sync_result.get('added_to_vector_store', 0)} transactions to vector store")
        logger.info(f"Vector store now has {sync_result.get('vector_store_after', 0)} transactions")
        
    except ImportError as e:
        logger.error(f"Error importing required modules: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error syncing with vector store: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 