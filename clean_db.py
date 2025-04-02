#!/usr/bin/env python
"""
Database Cleanup Utility for Finance LLM App

This script removes redundant databases and ensures a consistent database structure.
It will:
1. Check for multiple databases in different locations
2. Retain only the main database at the project root
3. Optionally reset the vector store to ensure consistency

Usage:
    python clean_db.py [--reset-vectors]
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("db_cleanup")

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Clean up redundant databases and ensure consistency")
    parser.add_argument('--reset-vectors', action='store_true', help='Reset vector store after cleaning')
    parser.add_argument('--force', action='store_true', help='Force cleanup without confirmation')
    args = parser.parse_args()
    
    logger.info("Starting database cleanup process")
    
    # Define paths
    root_db_path = Path("./finance.db")
    processed_db_path = Path("./data/processed/finance.db")
    
    # Check which databases exist
    root_db_exists = root_db_path.exists()
    processed_db_exists = processed_db_path.exists()
    
    logger.info(f"Root database status: {'exists' if root_db_exists else 'does not exist'}")
    logger.info(f"Processed directory database status: {'exists' if processed_db_exists else 'does not exist'}")
    
    # Determine action based on what exists
    if root_db_exists and processed_db_exists:
        logger.info("Both databases exist - need to remove redundancy")
        
        # Get file sizes and modification times
        root_size = root_db_path.stat().st_size
        processed_size = processed_db_path.stat().st_size
        root_mtime = root_db_path.stat().st_mtime
        processed_mtime = processed_db_path.stat().st_mtime
        
        logger.info(f"Root database: {root_size / 1024:.1f} KB, modified {format_time(root_mtime)}")
        logger.info(f"Processed database: {processed_size / 1024:.1f} KB, modified {format_time(processed_mtime)}")
        
        # Determine which is newer/larger
        if processed_mtime > root_mtime:
            newer_db = "processed"
            logger.info("Processed database is newer")
        else:
            newer_db = "root"
            logger.info("Root database is newer")
            
        if processed_size > root_size:
            larger_db = "processed"
            logger.info("Processed database is larger")
        else:
            larger_db = "root"
            logger.info("Root database is larger")
        
        # Recommend action
        if newer_db == larger_db == "processed":
            recommended_action = "keep_processed"
            logger.info("Recommendation: Keep processed database (newer and larger)")
        elif newer_db == larger_db == "root":
            recommended_action = "keep_root"
            logger.info("Recommendation: Keep root database (newer and larger)")
        else:
            # If one is newer but the other is larger, prefer the newer one
            if newer_db == "processed":
                recommended_action = "keep_processed"
                logger.info("Recommendation: Keep processed database (newer)")
            else:
                recommended_action = "keep_root"
                logger.info("Recommendation: Keep root database (newer)")
        
        # Get confirmation unless force flag is used
        if not args.force:
            if recommended_action == "keep_processed":
                prompt = "Replace root database with processed database? (y/n): "
            else:
                prompt = "Delete processed database and keep root database? (y/n): "
                
            response = input(prompt)
            confirmed = response.lower() in ['y', 'yes']
        else:
            confirmed = True
        
        # Execute the action
        if confirmed:
            if recommended_action == "keep_processed":
                logger.info("Copying processed database to root location")
                shutil.copy2(processed_db_path, root_db_path)
                logger.info("Removing processed database to avoid redundancy")
                os.remove(processed_db_path)
            else:
                logger.info("Removing processed database to avoid redundancy")
                os.remove(processed_db_path)
        else:
            logger.info("Action cancelled by user")
            return
            
    elif root_db_exists and not processed_db_exists:
        logger.info("Only root database exists - no cleanup needed")
    elif not root_db_exists and processed_db_exists:
        logger.info("Only processed database exists - moving to root location")
        logger.info("Copying processed database to root location")
        processed_db_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(processed_db_path, root_db_path)
        logger.info("Removing processed database to avoid redundancy")
        os.remove(processed_db_path)
    else:
        logger.warning("No databases found - nothing to clean up")
        return
    
    # Sync vector store if requested
    if args.reset_vectors:
        reset_vector_store()
    
    logger.info("Database cleanup completed")

def format_time(timestamp):
    """Format a timestamp into a readable string"""
    import datetime
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def reset_vector_store():
    """Reset and sync the vector store"""
    logger.info("Resetting and syncing vector store...")
    
    try:
        # Import required modules
        from src.embedding.vector_store import VectorStore
        from src.retrieval.retrieval_system import FinanceRetrieval
        
        # Initialize components
        vector_store = VectorStore()
        retrieval = FinanceRetrieval(vector_store=vector_store)
        
        # Reset vector store
        logger.info("Resetting vector store...")
        reset_result = vector_store.reset_vector_store()
        if reset_result.get("success", False):
            logger.info("Vector store reset completed successfully")
        else:
            logger.error(f"Vector store reset failed: {reset_result.get('error', 'Unknown error')}")
            return
            
        # Sync with vector store
        logger.info("Syncing database with vector store...")
        sync_result = retrieval.sync_database_with_vector_store()
        
        logger.info(f"Added {sync_result.get('added_to_vector_store', 0)} transactions to vector store")
        logger.info(f"Vector store now has {sync_result.get('vector_store_after', 0)} transactions")
        
    except ImportError as e:
        logger.error(f"Error importing required modules: {e}")
    except Exception as e:
        logger.error(f"Error syncing with vector store: {e}")

if __name__ == "__main__":
    main() 