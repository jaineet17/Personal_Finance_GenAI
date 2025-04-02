import os
import sys
import pandas as pd
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
import logging

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data_processing.data_sources import (
    ChaseCardImporter, 
    ChaseAccountImporter, 
    GenericBankImporter
)
from src.data_processing.data_standardizer import DataStandardizer
from src.embedding.vector_store import VectorStore

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / "data" / "processing.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("unified_processor")

class UnifiedProcessor:
    """Unified processor that handles importing, standardizing, and storing financial data"""

    def __init__(self):
        """Initialize the unified processor"""
        self.raw_data_path = Path(os.getenv("RAW_DATA_PATH", "./data/raw"))
        self.processed_data_path = Path(os.getenv("PROCESSED_DATA_PATH", "./data/processed"))
        self.vector_db_path = Path(os.getenv("VECTOR_DB_PATH", "./data/vectors/chroma"))

        # Ensure directories exist
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)

        # Database path
        self.db_path = self.processed_data_path / "finance.db"

        # Initialize components
        self.standardizer = DataStandardizer()
        self.vector_store = VectorStore()

        # Create data source importers
        self.importers = {
            "chase_card": ChaseCardImporter,
            "chase_account": ChaseAccountImporter,
            "generic": GenericBankImporter
        }

    def detect_source_type(self, file_path):
        """Detect the type of data source from the file"""
        file_name = Path(file_path).name.lower()

        if "chase" in file_name and ("activity" in file_name or ".csv" in file_name):
            # Check if it's a Chase credit card or bank account file
            try:
                # Read first few lines to determine
                with open(file_path, 'r') as f:
                    header = f.readline().lower()

                if "transaction date" in header or "post date" in header:
                    return "chase_card"
                elif "details" in header or "posting date" in header:
                    return "chase_account"
            except Exception as e:
                logger.warning(f"Error detecting Chase file type: {e}")

        # Default to generic importer
        return "generic"

    def process_file(self, file_path, source_type=None, account_name=None):
        """Process a single file"""
        logger.info(f"Processing file: {file_path}")

        try:
            # Detect source type if not provided
            if source_type is None:
                source_type = self.detect_source_type(file_path)
                logger.info(f"Detected source type: {source_type}")

            # Create appropriate importer
            importer_class = self.importers.get(source_type, GenericBankImporter)
            importer = importer_class(account_name=account_name)

            # Import data
            df = importer.import_data(file_path)
            logger.info(f"Imported {len(df)} transactions")

            # Anonymize data
            df_anon = importer.anonymize_data(df)
            logger.info("Anonymized data")

            # Standardize data
            df_std = self.standardizer.standardize_all(df_anon)
            logger.info("Standardized data")

            # Save to database
            num_saved = self.save_to_database(df_std)
            logger.info(f"Saved {num_saved} transactions to database")

            # Save to vector store
            embedding_success = self.save_to_vector_store(df_std)
            if embedding_success:
                logger.info("Saved to vector store")
            else:
                logger.warning("Failed to save to vector store")

            return num_saved, embedding_success

        except Exception as e:
            logger.error(f"Error processing file: {e}", exc_info=True)
            return 0, False

    def save_to_database(self, df):
        """Save processed data to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Select only the columns needed for the transactions table
            # Adjust the column list based on your database schema
            transaction_columns = [
                'date_iso', 'description', 'merchant', 'merchant_anon', 
                'amount', 'category', 'subcategory', 'account', 'account_id',
                'is_recurring', 'is_transfer', 'is_unusual', 'notes'
            ]

            # Select only columns that exist in the DataFrame
            valid_columns = [col for col in transaction_columns if col in df.columns]

            # Create a mapping of DataFrame columns to database columns
            column_mapping = {
                'date_iso': 'date',
                'merchant_anon': 'merchant',
                'is_unusual': 'is_outlier'
            }

            # Rename columns to match database schema
            df_to_save = df[valid_columns].copy()
            rename_columns = {k: v for k, v in column_mapping.items() if k in df_to_save.columns}
            df_to_save = df_to_save.rename(columns=rename_columns)

            # Save to transactions table
            df_to_save.to_sql('transactions', conn, if_exists='append', index=False)

            # Create monthly aggregate data
            self.save_monthly_aggregates(df, conn)

            # Close connection
            conn.close()

            return len(df)

        except Exception as e:
            logger.error(f"Error saving to database: {e}", exc_info=True)
            return 0

    def save_monthly_aggregates(self, df, conn):
        """Save monthly aggregate data to the database"""
        try:
            # Group by month_year and category
            monthly_agg = df.groupby(['month_year', 'category']).agg({
                'amount': 'sum',
                'date': 'count'
            }).reset_index()

            # Rename columns
            monthly_agg = monthly_agg.rename(columns={
                'date': 'transaction_count',
                'amount': 'total_amount'
            })

            # Add year and month columns
            monthly_agg['year'] = monthly_agg['month_year'].str[:4].astype(int)
            monthly_agg['month'] = monthly_agg['month_year'].str[5:].astype(int)

            # Save to monthly_aggregates table
            monthly_agg.to_sql('monthly_aggregates', conn, if_exists='append', index=False)

        except Exception as e:
            logger.error(f"Error saving monthly aggregates: {e}", exc_info=True)

    def save_to_vector_store(self, df, collection_name="finance_transactions"):
        """Save transactions to vector store for embeddings"""
        try:
            # Create collection
            self.vector_store.create_collection(collection_name)

            # Create unique IDs for each transaction
            transaction_ids = [f"tx_{i}" for i in range(len(df))]

            # Prepare transaction texts and metadata
            transaction_texts = df['rich_description'].tolist()

            # Create metadata for each transaction
            metadata_list = []
            for _, row in df.iterrows():
                metadata = {
                    'date': row['date_iso'],
                    'amount': str(row['amount']),
                    'category': row['category'],
                    'account': row['account'],
                    'merchant': row['merchant_anon'] if 'merchant_anon' in row else "",
                    'is_recurring': str(row.get('is_recurring', 0)),
                    'is_transfer': str(row.get('is_transfer', 0))
                }
                metadata_list.append(metadata)

            # Add to vector store
            self.vector_store.add_transactions(
                collection_name,
                transaction_ids,
                transaction_texts,
                metadata_list=metadata_list
            )

            return True

        except Exception as e:
            logger.error(f"Error saving to vector store: {e}", exc_info=True)
            return False

    def process_all_files(self, file_configs=None):
        """Process all files in the raw data directory or a specific list of files"""
        if file_configs is None:
            # Process all files in the raw data directory
            file_paths = list(self.raw_data_path.glob('*.csv'))
            file_configs = [{'file_path': str(path)} for path in file_paths]

        total_transactions = 0
        successful_files = 0

        for config in file_configs:
            file_path = config['file_path']
            source_type = config.get('source_type')
            account_name = config.get('account_name')

            num_saved, success = self.process_file(file_path, source_type, account_name)

            total_transactions += num_saved
            if num_saved > 0 and success:
                successful_files += 1

        logger.info(f"Processed {len(file_configs)} files, {successful_files} successful")
        logger.info(f"Total transactions processed: {total_transactions}")

        return total_transactions, successful_files


if __name__ == "__main__":
    # Example usage
    processor = UnifiedProcessor()

    # Define files to process
    file_configs = [
        {
            'file_path': 'data/raw/Chase0435_Activity20240301_20250331_20250331.CSV',
            'source_type': 'chase_card',
            'account_name': 'Chase Credit Card #1'
        },
        {
            'file_path': 'data/raw/Chase1707_Activity_20250331.CSV',
            'source_type': 'chase_account',
            'account_name': 'Chase Bank Account #1'
        },
        {
            'file_path': 'data/raw/activity.csv',
            'source_type': 'generic',
            'account_name': 'Bank Account #2'
        }
    ]

    # Process all files
    total, successful = processor.process_all_files(file_configs)

    print(f"Processed {total} transactions from {successful} files") 