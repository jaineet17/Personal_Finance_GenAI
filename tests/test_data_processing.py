import os
import sys
import unittest
import pandas as pd
import sqlite3
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_processing.processor import DataProcessor
from src.db.database import Database

class TestDataProcessing(unittest.TestCase):
    """Test data processing and database components"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_data_dir = Path(project_root) / "tests" / "test_data"
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Create test raw data
        self.create_test_data()
        
        # Initialize processor with test data
        self.processor = DataProcessor(
            input_dir=str(self.test_data_dir / "raw"),
            output_dir=str(self.test_data_dir / "processed")
        )
        
        # Initialize test database
        self.db_path = self.test_data_dir / "test_finance.db"
        if self.db_path.exists():
            self.db_path.unlink()
        self.db = Database(f"sqlite:///{self.db_path}")
        
    def create_test_data(self):
        """Create test data files for processing"""
        # Create raw data directory
        raw_dir = self.test_data_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        
        # Create processed data directory
        processed_dir = self.test_data_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        # Create test CSV with sample transactions
        test_df = pd.DataFrame({
            'date': ['2023-01-15', '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19'],
            'description': ['STARBUCKS', 'AMAZON.COM', 'UBER', 'NETFLIX', 'ACME CORP PAYROLL'],
            'amount': [-4.85, -67.49, -24.35, -15.99, 2500.00],
            'category': ['Food', 'Shopping', 'Transportation', 'Entertainment', 'Income'],
            'account': ['Checking', 'Credit Card', 'Credit Card', 'Credit Card', 'Checking']
        })
        
        test_df.to_csv(raw_dir / "test_transactions.csv", index=False)
        
    def tearDown(self):
        """Clean up after tests"""
        # Close database connection
        if hasattr(self, 'db') and self.db.engine:
            self.db.engine.dispose()
            
    def test_data_completeness(self):
        """Test data completeness after processing"""
        # Process the test data
        self.processor.process_files()
        
        # Load raw and processed data
        raw_data = pd.read_csv(self.test_data_dir / "raw" / "test_transactions.csv")
        processed_files = list((self.test_data_dir / "processed").glob("*.csv"))
        
        # Ensure at least one processed file exists
        self.assertTrue(len(processed_files) > 0, "No processed files found")
        
        # Load the processed data
        processed_data = pd.read_csv(processed_files[0])
        
        # Check row counts match
        self.assertEqual(len(raw_data), len(processed_data), 
                         "Row counts don't match between raw and processed data")
        
    def test_data_type_validation(self):
        """Test data types are correct after processing"""
        # Process the test data
        self.processor.process_files()
        
        # Load processed data
        processed_files = list((self.test_data_dir / "processed").glob("*.csv"))
        processed_data = pd.read_csv(processed_files[0])
        
        # Check date format (attempt to convert to datetime)
        try:
            pd.to_datetime(processed_data['date'], format='%Y-%m-%d')
            date_format_valid = True
        except:
            date_format_valid = False
            
        self.assertTrue(date_format_valid, "Date format validation failed")
        
        # Check amount precision
        amount_col = processed_data['amount']
        decimal_precision = max([len(str(x).split('.')[-1]) for x in amount_col if '.' in str(x)])
        self.assertLessEqual(decimal_precision, 2, 
                            f"Amount decimal precision ({decimal_precision}) exceeds 2 digits")
        
    def test_database_queries(self):
        """Test basic database queries"""
        # Process and import test data
        self.processor.process_files()
        processed_files = list((self.test_data_dir / "processed").glob("*.csv"))
        
        # Create tables and import data
        self.db.create_tables()
        self.db.import_transactions(str(processed_files[0]))
        
        # Test querying all transactions
        all_txs = self.db.query_transactions()
        self.assertEqual(len(all_txs), 5, "Failed to retrieve all transactions")
        
        # Test querying with date filter
        filtered_txs = self.db.query_transactions(start_date="2023-01-16", end_date="2023-01-18")
        self.assertEqual(len(filtered_txs), 3, "Date filtering failed")
        
        # Test querying with category filter
        category_txs = self.db.query_transactions(category="Food")
        self.assertEqual(len(category_txs), 1, "Category filtering failed")
        
    def test_aggregation_calculations(self):
        """Test aggregation calculations in the database"""
        # Process and import test data
        self.processor.process_files()
        processed_files = list((self.test_data_dir / "processed").glob("*.csv"))
        
        # Create tables and import data
        self.db.create_tables()
        self.db.import_transactions(str(processed_files[0]))
        
        # Calculate total expenses manually
        total_expenses = -4.85 - 67.49 - 24.35 - 15.99
        
        # Query the database for total expenses
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(amount) FROM transactions WHERE amount < 0")
        db_expenses = cursor.fetchone()[0]
        
        # Compare calculated vs. database values (allow for small float differences)
        self.assertAlmostEqual(abs(total_expenses), abs(db_expenses), places=2, 
                              msg="Total expenses calculation mismatch")
        
        # Test monthly aggregation
        cursor.execute("""
            SELECT strftime('%Y-%m', date) as month, SUM(amount) 
            FROM transactions 
            GROUP BY month
        """)
        monthly_totals = cursor.fetchall()
        self.assertEqual(len(monthly_totals), 1, "Monthly aggregation failed")
        
        # Close connection
        conn.close()

if __name__ == '__main__':
    unittest.main() 