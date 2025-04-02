import os
import pandas as pd
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    """
    Database class for managing financial transaction data.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the database connection.
        
        Args:
            db_path: SQLite connection string (default: from env or "./finance.db")
        """
        # First try to get db_path from environment variable
        if db_path is None:
            db_path = os.getenv("DB_PATH", "./finance.db")
            
        # Log the actual path being used
        logger.info(f"Using database path: {db_path}")
            
        # Handle SQLite connection string
        if db_path.startswith("sqlite:///"):
            self.db_path = db_path
            file_path = db_path[len("sqlite:///"):]
            self.file_path = file_path
        else:
            # If it's just a file path, convert to SQLite connection string
            self.db_path = f"sqlite:///{db_path}"
            self.file_path = db_path
            
        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(self.file_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
            
        # Initialize engine and connection
        self.engine = None
        self.conn = None
        self._connect()
        
        logger.info(f"Initialized database with path: {self.db_path}")
    
    def _connect(self):
        """
        Connect to the SQLite database.
        """
        try:
            # Use check_same_thread=False to allow usage in different threads
            self.conn = sqlite3.connect(self.file_path, check_same_thread=False)
            logger.info(f"Connected to database: {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    def create_tables(self):
        """
        Create the necessary tables in the database.
        """
        try:
            cursor = self.conn.cursor()
            
            # Create transactions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                description TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT,
                account TEXT,
                merchant TEXT,
                month TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create categories table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                parent_category TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create accounts table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                type TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create monthly_aggregates table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS monthly_aggregates (
                month TEXT NOT NULL,
                category TEXT NOT NULL,
                total_amount REAL NOT NULL,
                count INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (month, category)
            )
            ''')
            
            # Create indices for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_account ON transactions(account)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_month ON transactions(month)')
            
            self.conn.commit()
            logger.info("Database tables created successfully")
        
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            raise
    
    def import_transactions(self, csv_path: str) -> int:
        """
        Import transactions from a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Number of imported transactions
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            logger.info(f"Importing {len(df)} transactions from {csv_path}")
            
            # Ensure required columns exist
            required_cols = ['date', 'description', 'amount']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' missing from CSV")
            
            # Extract month from date for aggregation
            df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
            
            # Add merchant column if not present
            if 'merchant' not in df.columns:
                df['merchant'] = df['description'].apply(self._extract_merchant)
            
            # Ensure ID column exists
            if 'id' not in df.columns:
                df['id'] = [f"tx_{i+1:06d}" for i in range(len(df))]
            
            # Convert to list of dictionaries
            transactions = df.to_dict('records')
            
            # Insert into database
            cursor = self.conn.cursor()
            
            # Prepare column names and placeholders
            columns = list(transactions[0].keys())
            placeholders = ", ".join(["?" for _ in columns])
            column_str = ", ".join(columns)
            
            # Prepare insert query
            query = f"INSERT OR REPLACE INTO transactions ({column_str}) VALUES ({placeholders})"
            
            # Execute batch insert
            values = []
            for tx in transactions:
                row = [tx.get(col) for col in columns]
                values.append(tuple(row))
            
            cursor.executemany(query, values)
            self.conn.commit()
            
            # Update aggregates
            self._update_aggregates()
            
            logger.info(f"Successfully imported {len(transactions)} transactions")
            return len(transactions)
            
        except Exception as e:
            logger.error(f"Error importing transactions: {str(e)}")
            raise
    
    def _extract_merchant(self, description: str) -> str:
        """
        Extract merchant name from transaction description.
        
        Args:
            description: Transaction description
            
        Returns:
            Extracted merchant name
        """
        # Simple extraction - take first word and remove non-alphanumeric chars
        parts = description.strip().split()
        if not parts:
            return "UNKNOWN"
        
        merchant = parts[0]
        # Remove transaction numbers, dates, etc.
        merchant = ''.join(c for c in merchant if c.isalnum() or c.isspace())
        return merchant.upper()
    
    def _update_aggregates(self):
        """
        Update monthly aggregates based on transaction data.
        """
        cursor = self.conn.cursor()
        
        # Clear existing aggregates
        cursor.execute("DELETE FROM monthly_aggregates")
        
        # Calculate new aggregates
        cursor.execute('''
        INSERT INTO monthly_aggregates (month, category, total_amount, count)
        SELECT 
            month, 
            COALESCE(category, 'Uncategorized') as category, 
            SUM(amount) as total_amount, 
            COUNT(*) as count
        FROM transactions
        GROUP BY month, category
        ''')
        
        self.conn.commit()
        logger.info("Monthly aggregates updated successfully")
    
    def query_transactions(self, 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None, 
                          category: Optional[str] = None,
                          account: Optional[str] = None,
                          min_amount: Optional[float] = None,
                          max_amount: Optional[float] = None,
                          limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Query transactions with filters.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            category: Category filter
            account: Account filter
            min_amount: Minimum amount
            max_amount: Maximum amount
            limit: Maximum number of results
            
        Returns:
            List of transactions as dictionaries
        """
        try:
            query = "SELECT * FROM transactions WHERE 1=1"
            params = []
            
            # Apply filters
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
                
            # Handle month-specific queries more effectively
            # If start_date and end_date are in the same month, add a direct month filter
            if start_date and end_date and start_date[:7] == end_date[:7]:
                month = start_date[:7]  # YYYY-MM format
                query += " AND month = ?"
                params.append(month)
            
            if category:
                query += " AND category = ?"
                params.append(category)
            
            if account:
                query += " AND account = ?"
                params.append(account)
            
            if min_amount is not None:
                query += " AND amount >= ?"
                params.append(min_amount)
            
            if max_amount is not None:
                query += " AND amount <= ?"
                params.append(max_amount)
            
            # Add order and limit
            query += " ORDER BY date DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            # Log the query for debugging
            logger.debug(f"SQL query: {query} with params {params}")
            
            # Execute query
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Convert to list of dictionaries
            result = []
            for row in cursor.fetchall():
                record = dict(zip(columns, row))
                result.append(record)
            
            logger.info(f"Query returned {len(result)} transactions")
            return result
            
        except Exception as e:
            logger.error(f"Error querying transactions: {str(e)}")
            raise
            
    def execute_raw_query(self, query: str, params: List = None) -> List[Dict[str, Any]]:
        """
        Execute a raw SQL query with optional parameters.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of records as dictionaries
        """
        try:
            # Default to empty list if params is None
            if params is None:
                params = []
                
            logger.debug(f"Executing raw SQL: {query}")
            
            # Execute query
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Convert to list of dictionaries
            result = []
            for row in cursor.fetchall():
                record = dict(zip(columns, row))
                result.append(record)
            
            logger.info(f"Raw query returned {len(result)} records")
            return result
            
        except Exception as e:
            logger.error(f"Error executing raw query: {str(e)}")
            raise
    
    def close(self):
        """
        Close the database connection.
        """
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def query_transactions_by_date(self, start_date=None, end_date=None, limit=None, category=None):
        """Query transactions by date range
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            limit: Maximum number of transactions to return
            category: Filter by category
            
        Returns:
            List of transaction dictionaries
        """
        try:
            query = "SELECT * FROM transactions WHERE 1=1"
            params = []
            
            # Format dates if they are date objects
            if start_date:
                if isinstance(start_date, date):
                    start_date = start_date.strftime("%Y-%m-%d")
                query += " AND date >= ?"
                params.append(start_date)
                
            if end_date:
                if isinstance(end_date, date):
                    end_date = end_date.strftime("%Y-%m-%d")
                query += " AND date <= ?"
                params.append(end_date)
                
            if category:
                query += " AND category = ?"
                params.append(category)
                
            query += " ORDER BY date DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
                
            transactions = self.execute_raw_query(query, params)
            return [dict(zip(['id', 'date', 'description', 'amount', 'category'], t)) for t in transactions]
        except Exception as e:
            logger.error(f"Error querying transactions: {e}")
            return []

    def get_top_spending_categories(self, start_date=None, end_date=None, limit=5):
        """Get top spending categories by amount
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            limit: Maximum number of categories to return
            
        Returns:
            List of (category, total_amount) tuples
        """
        try:
            query = """
                SELECT category, SUM(amount) as total
                FROM transactions
                WHERE amount < 0
            """
            params = []
            
            # Format dates if they are date objects
            if start_date:
                if isinstance(start_date, date):
                    start_date = start_date.strftime("%Y-%m-%d")
                query += " AND date >= ?"
                params.append(start_date)
                
            if end_date:
                if isinstance(end_date, date):
                    end_date = end_date.strftime("%Y-%m-%d")
                query += " AND date <= ?"
                params.append(end_date)
            
            query += """
                GROUP BY category
                ORDER BY total ASC
                LIMIT ?
            """
            params.append(limit)
                
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            
            # Get results directly from cursor to avoid column name issues
            results = cursor.fetchall()
            return [(category, float(amount)) for category, amount in results]
        except Exception as e:
            logger.error(f"Error getting top spending categories: {e}")
            return [] 