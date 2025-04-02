import os
import sys
import sqlite3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from other modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.embedding.vector_store import VectorStore

load_dotenv()

class TransactionProcessor:
    def __init__(self):
        """Initialize the transaction processor"""
        self.raw_data_path = os.getenv("RAW_DATA_PATH", "./data/raw")
        self.processed_data_path = os.getenv("PROCESSED_DATA_PATH", "./data/processed")
        self.db_path = Path(self.processed_data_path) / "finance.db"
        
        # Create directories if they don't exist
        Path(self.raw_data_path).mkdir(parents=True, exist_ok=True)
        Path(self.processed_data_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize vector store
        self.vector_store = VectorStore()
        
    def process_csv(self, file_path, date_col="Date", desc_col="Description", 
                    amount_col="Amount", category_col=None, account_col=None):
        """Process a CSV file of financial transactions
        
        Args:
            file_path (str): Path to the CSV file
            date_col (str): Name of the date column
            desc_col (str): Name of the description column
            amount_col (str): Name of the amount column
            category_col (str, optional): Name of the category column
            account_col (str, optional): Name of the account column
            
        Returns:
            pd.DataFrame: Processed transactions
        """
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Rename columns to standard format
            column_mapping = {
                date_col: "date",
                desc_col: "description",
                amount_col: "amount"
            }
            
            if category_col:
                column_mapping[category_col] = "category"
            
            if account_col:
                column_mapping[account_col] = "account"
                
            # Rename only the columns that exist
            valid_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df.rename(columns=valid_columns)
            
            # Ensure required columns exist
            required_columns = ["date", "description", "amount"]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in the CSV file")
            
            # Add missing optional columns
            if "category" not in df.columns:
                df["category"] = "Uncategorized"
            
            if "account" not in df.columns:
                df["account"] = "Default Account"
                
            # Add notes column if it doesn't exist
            if "notes" not in df.columns:
                df["notes"] = ""
            
            # Standardize date format
            df["date"] = pd.to_datetime(df["date"]).dt.strftime('%Y-%m-%d')
            
            # Ensure amount is numeric
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            
            # Drop rows with invalid amounts
            df = df.dropna(subset=["amount"])
            
            return df
        
        except Exception as e:
            print(f"Error processing CSV file: {e}")
            return None
    
    def save_to_database(self, df):
        """Save processed transactions to the SQLite database
        
        Args:
            df (pd.DataFrame): DataFrame of processed transactions
            
        Returns:
            int: Number of transactions saved
        """
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            
            # Save transactions
            df[["date", "description", "category", "amount", "account", "notes"]].to_sql(
                "transactions", 
                conn, 
                if_exists="append", 
                index=False
            )
            
            # Close the connection
            conn.close()
            
            return len(df)
        
        except Exception as e:
            print(f"Error saving to database: {e}")
            return 0
    
    def embed_transactions(self, df, collection_name="finance_transactions"):
        """Create embeddings for transactions and store in vector database
        
        Args:
            df (pd.DataFrame): DataFrame of processed transactions
            collection_name (str): Name of the vector collection
            
        Returns:
            bool: Success status
        """
        try:
            # Create unique IDs for each transaction
            transaction_ids = [f"tx_{i}" for i in range(len(df))]
            
            # Prepare transaction texts and metadata
            transaction_texts = df["description"].tolist()
            
            metadata_list = []
            for _, row in df.iterrows():
                metadata = {
                    "date": row["date"],
                    "amount": str(row["amount"]),
                    "category": row["category"],
                    "account": row["account"]
                }
                metadata_list.append(metadata)
            
            # Store in vector database
            self.vector_store.add_transactions(
                collection_name,
                transaction_ids,
                transaction_texts,
                metadata_list=metadata_list
            )
            
            return True
        
        except Exception as e:
            print(f"Error embedding transactions: {e}")
            return False
            
    def process_file(self, file_path, **kwargs):
        """Process a file and store both in SQLite and vector database
        
        Args:
            file_path (str): Path to the CSV file
            **kwargs: Arguments for process_csv method
            
        Returns:
            tuple: (Number of transactions saved, Success status of embedding)
        """
        # Process CSV
        df = self.process_csv(file_path, **kwargs)
        if df is None or len(df) == 0:
            return 0, False
        
        # Save to database
        num_saved = self.save_to_database(df)
        
        # Embed transactions
        embedding_success = self.embed_transactions(df)
        
        return num_saved, embedding_success 