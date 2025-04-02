import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def create_database():
    """Initialize the SQLite database with required tables"""
    db_path = Path(os.getenv("PROCESSED_DATA_PATH", "./data/processed")) / "finance.db"
    
    # Create directory if it doesn't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create transactions table with enhanced fields
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        description TEXT NOT NULL,
        merchant TEXT,
        amount REAL NOT NULL,
        category TEXT,
        subcategory TEXT,
        account TEXT,
        account_id TEXT,
        is_recurring INTEGER DEFAULT 0,
        is_transfer INTEGER DEFAULT 0,
        is_outlier INTEGER DEFAULT 0,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create accounts table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS accounts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        account_id TEXT UNIQUE,
        type TEXT NOT NULL,
        balance REAL DEFAULT 0,
        currency TEXT DEFAULT 'USD',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create categories table with hierarchy
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        type TEXT NOT NULL,
        parent_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (parent_id) REFERENCES categories (id)
    )
    ''')
    
    # Create monthly aggregates table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS monthly_aggregates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        month_year TEXT NOT NULL,
        year INTEGER NOT NULL,
        month INTEGER NOT NULL,
        category TEXT NOT NULL,
        total_amount REAL NOT NULL,
        transaction_count INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(month_year, category)
    )
    ''')
    
    # Create recurring transactions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS recurring_transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        merchant TEXT NOT NULL,
        category TEXT,
        typical_amount REAL,
        frequency_days INTEGER,
        last_date TEXT,
        next_expected_date TEXT,
        account TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create merchant mapping table for name normalization
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS merchant_mapping (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_name TEXT NOT NULL,
        normalized_name TEXT NOT NULL,
        anonymized_name TEXT NOT NULL,
        category TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(original_name)
    )
    ''')
    
    # Create indices for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_merchant ON transactions(merchant)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_account ON transactions(account_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_monthly_aggregates_date ON monthly_aggregates(year, month)')
    
    # Insert default categories
    default_categories = [
        ('Food', 'expense', None),
        ('Travel', 'expense', None),
        ('Shopping', 'expense', None),
        ('Entertainment', 'expense', None),
        ('Health', 'expense', None),
        ('Housing', 'expense', None),
        ('Utilities', 'expense', None),
        ('Education', 'expense', None),
        ('Services', 'expense', None),
        ('Income', 'income', None),
        ('Transfer', 'transfer', None),
        ('Fees', 'expense', None),
        ('Other', 'expense', None)
    ]
    
    cursor.executemany('''
    INSERT OR IGNORE INTO categories (name, type, parent_id)
    VALUES (?, ?, ?)
    ''', default_categories)
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Database initialized at {db_path}")


def export_category_mapping_json():
    """Export category mapping to JSON file for standardization"""
    db_path = Path(os.getenv("PROCESSED_DATA_PATH", "./data/processed")) / "finance.db"
    json_path = Path(os.getenv("PROCESSED_DATA_PATH", "./data/processed")) / "category_mapping.json"
    
    # Create a default mapping
    default_mapping = {
        # Chase card categories
        "Food & Drink": "Food",
        "Groceries": "Food",
        "Dining": "Food",
        "Travel": "Travel",
        "Transportation": "Travel",
        "Gas": "Travel",
        "Shopping": "Shopping",
        "Entertainment": "Entertainment",
        "Health & Wellness": "Health",
        "Home": "Housing",
        "Bills & Utilities": "Utilities",
        "Education": "Education",
        "Professional Services": "Services",
        "Personal": "Personal",
        "Fees & Adjustments": "Fees",
        
        # Add more mappings as needed
        "Uncategorized": "Other"
    }
    
    if db_path.exists():
        try:
            # Connect to database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all subcategories from transactions
            cursor.execute('''
            SELECT DISTINCT subcategory, category 
            FROM transactions 
            WHERE subcategory IS NOT NULL
            ''')
            
            # Update mapping with actual data
            for subcategory, category in cursor.fetchall():
                if subcategory and category:
                    default_mapping[subcategory] = category
            
            conn.close()
        except Exception as e:
            print(f"Error querying database for categories: {e}")
    
    # Save mapping to JSON
    try:
        import json
        with open(json_path, 'w') as f:
            json.dump(default_mapping, f, indent=2)
        print(f"Category mapping exported to {json_path}")
    except Exception as e:
        print(f"Error exporting category mapping: {e}")


if __name__ == "__main__":
    create_database()
    export_category_mapping_json() 