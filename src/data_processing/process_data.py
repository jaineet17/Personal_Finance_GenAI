#!/usr/bin/env python

import os
import sys
import re
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

import pandas as pd
from src.data_processing.processor import DataProcessor
from src.db.database import Database
from src.embedding.vector_store import VectorStore
from src.embedding.embedding_pipeline import EmbeddingPipeline
import logging
import sqlite3

def main():
    """Process raw financial data and load it into the database"""
    print("Processing raw financial data...")
    
    # Set up paths
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    # Make sure directories exist
    raw_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    
    # Remove old processed CSV files to avoid data duplication
    for old_file in processed_dir.glob("processed_*.csv"):
        print(f"Removing old processed file: {old_file}")
        old_file.unlink()
    
    # Initialize processor
    processor = DataProcessor(
        input_dir=str(raw_dir),
        output_dir=str(processed_dir)
    )
    
    # Process the regular activity.csv file
    def process_activity_file():
        """Process the main activity.csv file manually to escape problematic characters"""
        activity_file = raw_dir / "activity.csv"
        if not activity_file.exists():
            print(f"Activity file not found: {activity_file}")
            return []
            
        output_file = processed_dir / f"processed_activity.csv"
        
        try:
            # Read CSV with proper escaping of quotes and special characters
            df = pd.read_csv(activity_file, quotechar='"', escapechar='\\')
            
            # Create a clean dataframe with only the columns we need
            clean_df = pd.DataFrame()
            clean_df['date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            clean_df['description'] = df['Description'].str.replace("'", "''")  # Escape single quotes for SQL
            clean_df['amount'] = df['Amount']
            clean_df['category'] = df['Category']
            clean_df['account'] = 'AmericanExpress'
            clean_df['id'] = [f"amex_{i+1:06d}" for i in range(len(df))]
            clean_df['month'] = pd.to_datetime(clean_df['date']).dt.strftime('%Y-%m')
            
            # Save as new CSV
            clean_df.to_csv(output_file, index=False, quoting=1)  # quoting=1 means quote all strings
            print(f"Processed activity.csv file, saved to {output_file}")
            return [output_file]
        except Exception as e:
            print(f"Error processing activity.csv: {e}")
            return []
    
    # Process the main activity file
    processed_activity_files = process_activity_file()
    
    # Manually process Chase .CSV files
    def process_chase_csv_files():
        """Process Chase bank files with .CSV extension"""
        uppercase_csv_files = list(raw_dir.glob("*.CSV"))
        print(f"Found {len(uppercase_csv_files)} uppercase .CSV files")
        
        processed_files = []
        for csv_file in uppercase_csv_files:
            print(f"Processing Chase CSV file: {csv_file}")
            try:
                # Create a properly formatted output file
                output_file = processed_dir / f"processed_{csv_file.stem.lower()}.csv"
                
                # Read the Chase files differently based on format
                if "Chase0435" in csv_file.name:
                    # Chase credit card format
                    df = pd.read_csv(csv_file)
                    
                    # Standardize column names
                    renamed_df = pd.DataFrame()
                    renamed_df['date'] = pd.to_datetime(df['Transaction Date']).dt.strftime('%Y-%m-%d')
                    renamed_df['description'] = df['Description'].str.replace("'", "''")  # Escape single quotes
                    renamed_df['amount'] = df['Amount']
                    renamed_df['category'] = df['Category']
                    renamed_df['account'] = 'Chase0435'
                    renamed_df['id'] = [f"chase0435_{i+1:06d}" for i in range(len(df))]
                    renamed_df['month'] = pd.to_datetime(renamed_df['date']).dt.strftime('%Y-%m')
                    
                    # Save processed file
                    renamed_df.to_csv(output_file, index=False, quoting=1)  # quoting=1 means quote strings
                    processed_files.append(output_file)
                    print(f"Processed Chase credit card file, saved to {output_file}")
                    
                elif "Chase1707" in csv_file.name:
                    # Chase bank account format - this has a different structure
                    try:
                        # First try direct parsing
                        df = pd.read_csv(csv_file)
                        
                        # Verify columns match expected format
                        expected_cols = ['Details', 'Posting Date', 'Description', 'Amount', 'Type', 'Balance', 'Check or Slip #']
                        if set(df.columns) != set(expected_cols):
                            # Try reading with skiprows if columns don't match
                            df = pd.read_csv(csv_file, skiprows=1)
                            # Rename columns with fixed names
                            df.columns = expected_cols
                    except:
                        # If both approaches fail, try manual parsing
                        with open(csv_file, 'r') as f:
                            lines = f.readlines()
                        
                        # Extract header row and data rows
                        header = lines[0].strip().split(',')
                        data_rows = [line.strip().split(',') for line in lines[1:]]
                        
                        # Create DataFrame
                        df = pd.DataFrame(data_rows, columns=header)
                    
                    # Create standardized DataFrame
                    renamed_df = pd.DataFrame()
                    renamed_df['date'] = df.iloc[:, 0]  # First column is the date
                    renamed_df['description'] = df.iloc[:, 2]  # Third column is description
                    renamed_df['amount'] = pd.to_numeric(df.iloc[:, 3])  # Fourth column is amount
                    renamed_df['category'] = df.iloc[:, 4]  # Fifth column is type/category
                    renamed_df['account'] = 'Chase1707'
                    renamed_df['id'] = [f"chase1707_{i+1:06d}" for i in range(len(df))]
                    
                    # Convert date format and add month
                    renamed_df['date'] = pd.to_datetime(renamed_df['date'], format='%m/%d/%Y', errors='coerce').dt.strftime('%Y-%m-%d')
                    renamed_df['month'] = pd.to_datetime(renamed_df['date'], errors='coerce').dt.strftime('%Y-%m')
                    
                    # Clean up description to remove problematic characters
                    renamed_df['description'] = renamed_df['description'].astype(str).str.replace("'", "''")
                    
                    # Filter out rows with invalid dates
                    renamed_df = renamed_df.dropna(subset=['date'])
                    
                    # Save processed file
                    renamed_df.to_csv(output_file, index=False, quoting=1)
                    processed_files.append(output_file)
                    print(f"Processed Chase bank account file, saved to {output_file}")
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
        
        return processed_files
    
    # Process Chase CSV files
    processed_chase_files = process_chase_csv_files()
    print(f"Processed {len(processed_chase_files)} Chase .CSV files")
    
    # Initialize database - use the main database in the project root
    db_path = "./finance.db"
    
    # Ask user if they want to clear existing data
    clear_db = input("Do you want to clear existing data in the database? (y/n): ")
    
    # Initialize database
    if clear_db.lower() == 'y':
        # Remove old database if it exists to start fresh
        db_file = Path(db_path)
        if db_file.exists():
            print(f"Removing old database: {db_file}")
            db_file.unlink()
        
        # Initialize a new database
        db = Database(db_path=db_path)
        
        # Create tables
        db.create_tables()
    else:
        # Use existing database
        print(f"Using existing database at {db_path}")
        db = Database(db_path=db_path)
    
    # Import processed files into database
    total_transactions = 0
    all_processed_files = processed_activity_files + processed_chase_files
    
    for file in all_processed_files:
        if file.exists():
            try:
                num_transactions = db.import_transactions(str(file))
                total_transactions += num_transactions
                print(f"Imported {num_transactions} transactions from {file}")
            except Exception as e:
                print(f"Error importing {file}: {e}")
    
    print(f"Processed {len(all_processed_files)} files")
    print(f"Imported a total of {total_transactions} transactions into the database")
    print(f"Database path: {db_path}")
    
    # Sync with vector store
    try:
        from src.embedding.vector_store import VectorStore
        from src.retrieval.retrieval_system import FinanceRetrieval
        
        print("Syncing database with vector store...")
        vector_store = VectorStore()
        retrieval = FinanceRetrieval(vector_store=vector_store)
        
        # Reset and sync vector store
        print("Resetting vector store...")
        reset_result = vector_store.reset_vector_store()
        print(f"Vector store reset completed: {reset_result['success']}")
        
        # Sync database with vector store
        print("Syncing database with vector store...")
        sync_result = retrieval.sync_database_with_vector_store()
        print(f"Vector store sync completed: {sync_result['success']}")
        print(f"Added {sync_result.get('added_to_vector_store', 0)} transactions to vector store")
        print(f"Vector store now has {sync_result.get('vector_store_after', 0)} transactions")
    except Exception as e:
        print(f"Error syncing with vector store: {e}")

def process_file_to_db(file_path: str, db: Database):
    """Process a single file and load it into the database
    
    Args:
        file_path: Path to the file to process
        db: Database instance to load the data into
        
    Returns:
        dict: Dictionary with processing results
    """
    import tempfile
    import pandas as pd
    from pathlib import Path
    import logging
    import os
    import sqlite3
    from src.embedding.vector_store import VectorStore
    from src.embedding.embedding_pipeline import EmbeddingPipeline
    
    logger = logging.getLogger("file-processor")
    
    file_path = Path(file_path)
    
    # Verify database connection
    try:
        # Test database connection
        if db.conn is None or not hasattr(db.conn, 'cursor'):
            logger.warning("Database connection invalid, reconnecting")
            db._connect()
            
        # Ensure tables exist
        db.create_tables()
    except sqlite3.Error as e:
        logger.error(f"SQLite error verifying database: {e}")
        # If there's a connection issue, create a new database instance
        logger.info("Creating new database connection")
        db = Database()
        db.create_tables()
    except Exception as e:
        logger.error(f"Error verifying database: {e}")
        # If there's a connection issue, create a new database instance
        logger.info("Creating new database connection")
        db = Database()
        db.create_tables()
    
    # Determine file extension
    file_ext = file_path.suffix.lower()
    
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create a DataProcessor for this file
        processor = DataProcessor(
            input_dir=str(file_path.parent),
            output_dir=str(temp_dir)
        )
        
        # Initialize results
        results = {
            "transactions_processed": 0,
            "categories_detected": set(),
            "date_range": ""
        }
        
        try:
            # Process based on file type
            if file_ext == '.csv':
                # For CSV files
                output_file = temp_dir / f"processed_{file_path.stem}.csv"
                
                # Read CSV with proper escaping
                df = pd.read_csv(file_path, quotechar='"', escapechar='\\', on_bad_lines='skip')
                
                # Check if this is a standard CSV or a specific format
                if "Transaction Date" in df.columns and "Description" in df.columns and "Amount" in df.columns:
                    # Looks like Chase format
                    clean_df = pd.DataFrame()
                    clean_df['date'] = pd.to_datetime(df['Transaction Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                    clean_df['description'] = df['Description'].str.replace("'", "''")  # Escape single quotes
                    clean_df['amount'] = df['Amount']
                    clean_df['category'] = df.get('Category', 'Unknown')
                    clean_df['account'] = 'Uploaded'
                    clean_df['id'] = [f"upload_{i+1:06d}" for i in range(len(df))]
                    clean_df['month'] = pd.to_datetime(clean_df['date'], errors='coerce').dt.strftime('%Y-%m')
                elif "Date" in df.columns and "Description" in df.columns and "Amount" in df.columns:
                    # Standard format
                    clean_df = pd.DataFrame()
                    clean_df['date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                    clean_df['description'] = df['Description'].str.replace("'", "''")  # Escape single quotes
                    clean_df['amount'] = df['Amount']
                    clean_df['category'] = df.get('Category', 'Unknown')
                    clean_df['account'] = 'Uploaded'
                    clean_df['id'] = [f"upload_{i+1:06d}" for i in range(len(df))]
                    clean_df['month'] = pd.to_datetime(clean_df['date'], errors='coerce').dt.strftime('%Y-%m')
                else:
                    # Try to infer columns
                    date_col = next((col for col in df.columns if "date" in col.lower()), df.columns[0])
                    description_col = next((col for col in df.columns if "descr" in col.lower() or "narr" in col.lower()), df.columns[1]) 
                    amount_col = next((col for col in df.columns if "amount" in col.lower() or "sum" in col.lower()), df.columns[2])
                    
                    clean_df = pd.DataFrame()
                    clean_df['date'] = pd.to_datetime(df[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
                    clean_df['description'] = df[description_col].astype(str).str.replace("'", "''")
                    clean_df['amount'] = df[amount_col]
                    clean_df['category'] = 'Unknown'  # Default category
                    clean_df['account'] = 'Uploaded'
                    clean_df['id'] = [f"upload_{i+1:06d}" for i in range(len(df))]
                    clean_df['month'] = pd.to_datetime(clean_df['date'], errors='coerce').dt.strftime('%Y-%m')
                
                # Filter out rows with invalid dates
                clean_df = clean_df.dropna(subset=['date'])
                
                # Save as new CSV
                clean_df.to_csv(output_file, index=False, quoting=1)
                
                # Log the path and confirm the file was created
                logger.info(f"Processed file saved to {output_file}")
                if not os.path.exists(output_file):
                    logger.error(f"Failed to create output file at {output_file}")
                    raise FileNotFoundError(f"Output file {output_file} not found")
                
                # Import into database
                logger.info(f"Importing transactions into database from {output_file}")
                try:
                    num_transactions = db.import_transactions(str(output_file))
                    logger.info(f"Imported {num_transactions} transactions into database")
                except sqlite3.Error as e:
                    logger.error(f"SQLite error during import: {e}")
                    # Try to recreate connection and retry
                    db._connect()
                    num_transactions = db.import_transactions(str(output_file))
                    logger.info(f"Retry succeeded: Imported {num_transactions} transactions into database")
                
                # Add to vector store for retrieval
                try:
                    logger.info("Indexing transactions in vector store")
                    vector_store = VectorStore()
                    embedding_pipeline = EmbeddingPipeline()
                    
                    # Convert dataframe to list of dictionaries
                    transactions = clean_df.to_dict('records')
                    
                    # Format transactions for vector store
                    formatted_transactions = []
                    for tx in transactions:
                        # Create text representation
                        tx_text = f"Transaction: {tx['description']} Amount: {tx['amount']} Category: {tx.get('category', 'Unknown')} Date: {tx['date']}"
                        
                        # Create numeric date for filtering
                        numeric_date = int(tx['date'].replace('-', '')) if 'date' in tx else 0
                        
                        # Format metadata
                        metadata = {
                            "id": tx.get('id', ''),
                            "date": tx.get('date', ''),
                            "description": tx.get('description', ''),
                            "amount": float(tx.get('amount', 0)),
                            "category": tx.get('category', 'Unknown'),
                            "account": tx.get('account', 'Unknown'),
                            "numeric_date": numeric_date
                        }
                        
                        formatted_transactions.append({
                            "id": tx.get('id', ''),
                            "text": tx_text,
                            "metadata": metadata
                        })
                    
                    # Add to vector store
                    if formatted_transactions:
                        vector_store.add_transactions(formatted_transactions)
                        logger.info(f"Added {len(formatted_transactions)} transactions to vector store")
                except Exception as e:
                    logger.error(f"Error indexing transactions in vector store: {e}")
                
                # Update results
                results["transactions_processed"] = num_transactions
                
                # Clean up categories to remove NaN values
                import math
                import numpy as np
                all_categories = clean_df['category'].unique()
                
                # Filter out NaN and None values
                valid_categories = [
                    cat for cat in all_categories 
                    if cat is not None 
                    and not (isinstance(cat, float) and (math.isnan(cat) or np.isnan(cat)))
                ]
                
                results["categories_detected"] = valid_categories
                
                # Determine date range
                min_date = clean_df['date'].min()
                max_date = clean_df['date'].max()
                results["date_range"] = f"{min_date} to {max_date}"
                
                return results
                
            else:
                # Handle other file types here if needed
                raise ValueError(f"Unsupported file type: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error processing file: {e}", exc_info=True)
            raise Exception(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main() 