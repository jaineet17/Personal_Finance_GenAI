#!/usr/bin/env python
"""
Vector Store Initialization Script

This script initializes the vector store with transaction data from the database.
It retrieves transactions, extracts categories, and adds them to the vector store.

Usage:
    python init_vector_store.py

This should be run whenever the database is updated or when the vector store needs to be rebuilt.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.db.database import Database
from src.embedding.vector_store import VectorStore

def get_categories_from_transactions(transactions):
    """Extract unique categories from transactions"""
    categories = set()
    for transaction in transactions:
        category = transaction.get('category')
        if category and isinstance(category, str) and category.strip():
            categories.add(category.strip())
    return list(categories)

def main():
    """Initialize the vector store with transactions from the database"""
    print("Initializing vector store with transactions from database...")
    
    # Initialize database and vector store
    db = Database()
    vector_store = VectorStore()
    
    # Get all transactions from the database
    print("Retrieving transactions from database...")
    transactions = db.query_transactions(limit=10000)  # Get up to 10,000 transactions
    
    if not transactions:
        print("No transactions found in the database.")
        return
    
    print(f"Found {len(transactions)} transactions in the database.")
    
    # Add transactions to vector store
    print("Adding transactions to vector store...")
    vector_store.add_transactions(transactions)
    print("Transactions added to vector store successfully.")
    
    # Add categories to vector store
    print("Adding categories to vector store...")
    categories = get_categories_from_transactions(transactions)
    if categories:
        vector_store.add_categories(categories)
        print(f"Added {len(categories)} categories to vector store.")
    else:
        print("No categories found in transactions.")
    
    # Add time periods to vector store
    print("Adding time periods to vector store...")
    vector_store.add_time_periods()
    print("Time periods added to vector store.")
    
    print("Vector store initialization complete!")

if __name__ == "__main__":
    main() 