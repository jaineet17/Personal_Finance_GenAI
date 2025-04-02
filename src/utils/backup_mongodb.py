#!/usr/bin/env python3
"""
MongoDB backup utility that exports MongoDB collections to JSON files
and uploads them to an S3 bucket.
"""

import os
import sys
import json
import datetime
import tempfile
import subprocess
import boto3
import pymongo
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get environment variables
MONGODB_URI = os.getenv("MONGODB_URI")
BACKUP_S3_BUCKET = os.getenv("BACKUP_S3_BUCKET")
BACKUP_S3_PREFIX = os.getenv("BACKUP_S3_PREFIX", "backups")

if not MONGODB_URI:
    print("Error: MONGODB_URI environment variable is required")
    sys.exit(1)

if not BACKUP_S3_BUCKET:
    print("Error: BACKUP_S3_BUCKET environment variable is required")
    sys.exit(1)

# Initialize S3 client
s3 = boto3.client('s3')

def backup_mongodb():
    """
    Main function to backup MongoDB collections to S3
    """
    # Create a timestamp for the backup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Connect to MongoDB
    client = pymongo.MongoClient(MONGODB_URI)
    
    # Get database name from connection string
    db_name = MONGODB_URI.split("/")[-1].split("?")[0]
    db = client[db_name]
    
    # Get all collections
    collections = db.list_collection_names()
    
    # Create a temporary directory for the backup files
    with tempfile.TemporaryDirectory() as temp_dir:
        backup_files = []
        
        print(f"Starting backup of database '{db_name}' with {len(collections)} collections")
        
        # Export each collection to a JSON file
        for collection_name in collections:
            output_file = os.path.join(temp_dir, f"{collection_name}.json")
            collection = db[collection_name]
            
            # Export collection to JSON
            with open(output_file, 'w') as f:
                documents = list(collection.find())
                # Convert ObjectId to string for JSON serialization
                for doc in documents:
                    if '_id' in doc and not isinstance(doc['_id'], str):
                        doc['_id'] = str(doc['_id'])
                        
                json.dump(documents, f, default=str)
            
            backup_files.append((output_file, collection_name))
            print(f"Exported collection '{collection_name}' with {len(documents)} documents")
        
        # Upload each file to S3
        for file_path, collection_name in backup_files:
            s3_key = f"{BACKUP_S3_PREFIX}/{timestamp}/{collection_name}.json"
            
            try:
                s3.upload_file(file_path, BACKUP_S3_BUCKET, s3_key)
                print(f"Uploaded {collection_name} to s3://{BACKUP_S3_BUCKET}/{s3_key}")
            except ClientError as e:
                print(f"Error uploading {collection_name} to S3: {e}")
                sys.exit(1)
    
    print(f"Backup completed successfully: {len(collections)} collections backed up to S3")
    return True

if __name__ == "__main__":
    print(f"Starting MongoDB backup to S3 bucket '{BACKUP_S3_BUCKET}'")
    success = backup_mongodb()
    if success:
        print("Backup completed successfully")
    else:
        print("Backup failed")
        sys.exit(1) 