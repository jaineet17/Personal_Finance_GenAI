"""
MongoDB initializer script for Finance RAG application.
This script creates necessary collections and indexes for MongoDB Atlas.
"""

import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, TEXT
import pymongo.errors

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get MongoDB connection string from environment
mongodb_uri = os.getenv("MONGODB_URI")
if not mongodb_uri:
    raise ValueError("MONGODB_URI environment variable is not set")

def initialize_mongodb():
    """Initialize MongoDB collections and indexes."""
    try:
        # Connect to MongoDB
        client = MongoClient(mongodb_uri)
        db_name = mongodb_uri.split('/')[-1].split('?')[0]
        db = client[db_name]

        logger.info(f"Connected to MongoDB database: {db_name}")

        # Create collections if they don't exist
        if "transactions" not in db.list_collection_names():
            db.create_collection("transactions")
            logger.info("Created 'transactions' collection")

        if "categories" not in db.list_collection_names():
            db.create_collection("categories")
            logger.info("Created 'categories' collection")

        if "vector_embeddings" not in db.list_collection_names():
            db.create_collection("vector_embeddings")
            logger.info("Created 'vector_embeddings' collection")

        if "query_cache" not in db.list_collection_names():
            db.create_collection("query_cache")
            logger.info("Created 'query_cache' collection")

        if "user_feedback" not in db.list_collection_names():
            db.create_collection("user_feedback")
            logger.info("Created 'user_feedback' collection")

        # Create indexes
        # Transactions collection indexes
        db.transactions.create_index([("date", ASCENDING)])
        db.transactions.create_index([("category", ASCENDING)])
        db.transactions.create_index([("amount", ASCENDING)])
        db.transactions.create_index([("description", TEXT)])
        logger.info("Created indexes for 'transactions' collection")

        # Categories collection indexes
        db.categories.create_index([("name", ASCENDING)], unique=True)
        logger.info("Created indexes for 'categories' collection")

        # Vector embeddings collection indexes
        db.vector_embeddings.create_index([("transaction_id", ASCENDING)], unique=True)
        logger.info("Created indexes for 'vector_embeddings' collection")

        # Query cache collection indexes
        db.query_cache.create_index([("query_hash", ASCENDING)], unique=True)
        db.query_cache.create_index([("created_at", ASCENDING)], expireAfterSeconds=86400)  # 24 hours TTL
        logger.info("Created indexes for 'query_cache' collection")

        # User feedback collection indexes
        db.user_feedback.create_index([("query_id", ASCENDING)])
        db.user_feedback.create_index([("created_at", ASCENDING)])
        logger.info("Created indexes for 'user_feedback' collection")

        logger.info("MongoDB initialization completed successfully")
        return True

    except pymongo.errors.ConnectionFailure as e:
        logger.error(f"MongoDB connection error: {str(e)}")
        return False
    except pymongo.errors.OperationFailure as e:
        logger.error(f"MongoDB operation error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error initializing MongoDB: {str(e)}")
        return False
    finally:
        if 'client' in locals():
            client.close()
            logger.info("MongoDB connection closed")

if __name__ == "__main__":
    if initialize_mongodb():
        logger.info("MongoDB setup completed successfully.")
    else:
        logger.error("Failed to set up MongoDB. Please check the logs for details.") 