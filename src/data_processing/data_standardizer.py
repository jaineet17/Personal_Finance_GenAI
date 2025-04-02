import pandas as pd
import numpy as np
from datetime import datetime
import re
import holidays
from pathlib import Path
import os
import json
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class DataStandardizer:
    """Class to standardize and enhance financial transaction data"""
    
    def __init__(self):
        """Initialize data standardizer"""
        # US holidays for holiday flag feature
        self.us_holidays = holidays.US()
        
        # Load category mapping if available
        self.category_mapping = self._load_category_mapping()
    
    def _load_category_mapping(self):
        """Load category mapping from file or return default"""
        mapping_file = Path(project_root) / "data" / "processed" / "category_mapping.json"
        
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                return json.load(f)
        
        # Default category mapping
        return {
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
    
    def standardize_date(self, df):
        """Convert all date formats to ISO standard (YYYY-MM-DD)"""
        df = df.copy()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Create ISO formatted date column
        df['date_iso'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # Extract additional date features
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_name'] = df['date'].dt.day_name()
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Week of month (1-5)
        df['week_of_month'] = (df['day'] - 1) // 7 + 1
        
        # Pay period indicator (assuming bi-weekly pay on 1st and 15th)
        df['pay_period'] = ((df['day'] >= 1) & (df['day'] <= 15)).astype(int) + 1
        
        # Holiday flag
        df['is_holiday'] = df['date_iso'].apply(lambda x: x in self.us_holidays).astype(int)
        
        return df
    
    def standardize_amount(self, df):
        """Standardize transaction amounts"""
        df = df.copy()
        
        # Ensure amount is numeric
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Ensure consistent sign convention (negative for expenses)
        # Some data sources might use positive for expenses, negative for income
        # Detect the convention and standardize
        if 'transaction_type' in df.columns:
            # If transaction_type exists, use it to determine sign
            expense_types = ['sale', 'purchase', 'payment', 'debit']
            income_types = ['credit', 'deposit', 'refund', 'credit']
            
            # Convert transaction_type to lowercase for case-insensitive comparison
            if not df['transaction_type'].empty:
                df['transaction_type_lower'] = df['transaction_type'].astype(str).str.lower()
                
                # For expense types, ensure amount is negative
                expense_mask = df['transaction_type_lower'].str.contains('|'.join(expense_types), na=False)
                df.loc[expense_mask, 'amount'] = -abs(df.loc[expense_mask, 'amount'])
                
                # For income types, ensure amount is positive
                income_mask = df['transaction_type_lower'].str.contains('|'.join(income_types), na=False)
                df.loc[income_mask, 'amount'] = abs(df.loc[income_mask, 'amount'])
                
                # Drop temporary column
                df = df.drop('transaction_type_lower', axis=1)
        
        # Round to standard precision (2 decimal places)
        df['amount'] = df['amount'].round(2)
        
        # Add absolute amount column
        df['amount_abs'] = df['amount'].abs()
        
        return df
    
    def map_categories(self, df):
        """Map institution-specific categories to standard categories"""
        df = df.copy()
        
        # If original_category exists, use it for mapping
        if 'original_category' in df.columns:
            # Create main category column
            df['category'] = df['original_category'].apply(
                lambda x: self.category_mapping.get(x, 'Other') if pd.notnull(x) else 'Other'
            )
            
            # Keep original category as subcategory
            df['subcategory'] = df['original_category']
        else:
            # If no original category, try to infer from description
            df['category'] = 'Other'
            df['subcategory'] = 'Uncategorized'
            
            # Infer categories from keywords in description
            self._infer_categories(df)
        
        return df
    
    def _infer_categories(self, df):
        """Infer categories based on description keywords"""
        # Define category keywords mapping
        category_keywords = {
            'Food': ['restaurant', 'cafe', 'coffee', 'doordash', 'grubhub', 'ubereats', 
                    'mcdonald', 'starbuck', 'grocery', 'food', 'pizza'],
            'Travel': ['airline', 'hotel', 'airbnb', 'uber', 'lyft', 'taxi', 'gas', 'travel', 
                      'flight', 'car rental', 'train', 'transit'],
            'Shopping': ['amazon', 'walmart', 'target', 'shop', 'store', 'market', 'retail', 
                        'clothing', 'apparel'],
            'Entertainment': ['netflix', 'hulu', 'spotify', 'disney', 'movie', 'theater', 
                             'entertainment', 'game', 'subscription'],
            'Health': ['doctor', 'medical', 'pharmacy', 'health', 'dental', 'fitness', 'gym'],
            'Housing': ['rent', 'mortgage', 'property', 'apartment', 'home', 'house'],
            'Utilities': ['electric', 'water', 'gas', 'internet', 'phone', 'utility', 'bill'],
            'Education': ['tuition', 'school', 'university', 'college', 'education', 'book', 'course'],
            'Income': ['salary', 'deposit', 'payroll', 'income', 'direct dep', 'payment received'],
            'Transfer': ['transfer', 'withdraw', 'deposit', 'zelle', 'venmo', 'paypal']
        }
        
        # Apply rules-based categorization
        for category, keywords in category_keywords.items():
            # Create pattern matching any of the keywords (case insensitive)
            pattern = '|'.join(keywords)
            mask = df['description'].str.lower().str.contains(pattern, na=False)
            df.loc[mask & (df['category'] == 'Other'), 'category'] = category
            df.loc[mask & (df['subcategory'] == 'Uncategorized'), 'subcategory'] = category
    
    def identify_recurring_transactions(self, df, time_window=90, min_occurrences=2):
        """Flag recurring transactions based on merchant and amount patterns"""
        df = df.copy()
        
        # Start with all transactions as non-recurring
        df['is_recurring'] = 0
        
        # Group by merchant_anon and check frequency
        merchant_counts = df['merchant_anon'].value_counts()
        recurring_merchants = merchant_counts[merchant_counts >= min_occurrences].index
        
        for merchant in recurring_merchants:
            # Get transactions for this merchant
            merchant_txns = df[df['merchant_anon'] == merchant].copy()
            
            if len(merchant_txns) >= min_occurrences:
                # Group by amount (rounded to nearest dollar to account for small variations)
                merchant_txns['amount_rounded'] = round(merchant_txns['amount'])
                
                # Count transactions with similar amounts
                amount_counts = merchant_txns['amount_rounded'].value_counts()
                recurring_amounts = amount_counts[amount_counts >= min_occurrences].index
                
                for amount in recurring_amounts:
                    # Get transactions with this merchant and similar amount
                    potential_recurring = merchant_txns[
                        merchant_txns['amount_rounded'] == amount
                    ]
                    
                    # If enough transactions, check for time patterns
                    if len(potential_recurring) >= min_occurrences:
                        # Sort by date
                        potential_recurring = potential_recurring.sort_values('date')
                        
                        # Calculate days between transactions
                        days_diff = potential_recurring['date'].diff().dt.days
                        
                        # If consistent interval (allowing for 3-day variance)
                        if days_diff.std() <= 3:
                            # Mark these transactions as recurring
                            recurring_indices = potential_recurring.index
                            df.loc[recurring_indices, 'is_recurring'] = 1
        
        # Remove temporary column if it was created
        if 'amount_rounded' in df.columns:
            df = df.drop('amount_rounded', axis=1)
            
        return df
    
    def identify_transfers(self, df):
        """Identify transfers between accounts"""
        df = df.copy()
        
        # Start with all transactions as non-transfers
        df['is_transfer'] = 0
        
        # Keywords that indicate transfers
        transfer_keywords = [
            'transfer', 'xfer', 'zelle', 'venmo', 'paypal', 'cash app',
            'withdrawal', 'deposit', 'atm', 'payment to', 'payment from'
        ]
        
        # Pattern for matching transfer keywords
        pattern = '|'.join(transfer_keywords)
        
        # Mark transfers based on description
        transfer_mask = df['description'].str.lower().str.contains(pattern, na=False)
        df.loc[transfer_mask, 'is_transfer'] = 1
        
        # Additional rule: Look for matching amounts with opposite signs close in time
        df = df.sort_values('date')
        
        # For each transaction, look for a matching opposite transaction within 3 days
        for idx, row in df.iterrows():
            if df.loc[idx, 'is_transfer'] == 0:  # Skip already identified transfers
                amount = row['amount']
                date = row['date']
                
                # Look for transactions with opposite amount within 3 days
                time_window_mask = (df['date'] >= date - pd.Timedelta(days=3)) & \
                                  (df['date'] <= date + pd.Timedelta(days=3))
                opposite_amount_mask = abs(df['amount'] + amount) < 0.1  # Allow for small differences
                
                matching_transfers = df[time_window_mask & opposite_amount_mask & (df.index != idx)]
                
                if len(matching_transfers) > 0:
                    # Mark both this transaction and the matching one as transfers
                    df.loc[idx, 'is_transfer'] = 1
                    df.loc[matching_transfers.index, 'is_transfer'] = 1
        
        return df
    
    def calculate_aggregated_features(self, df):
        """Calculate aggregated features like monthly averages and spending velocity"""
        df = df.copy()
        
        # Add month-year column for grouping
        df['month_year'] = df['date'].dt.strftime('%Y-%m')
        
        # Calculate monthly spending by category
        monthly_category = df.groupby(['month_year', 'category'])['amount'].sum().reset_index()
        
        # Pivot to get categories as columns
        monthly_pivot = monthly_category.pivot_table(
            index='month_year', 
            columns='category', 
            values='amount', 
            aggfunc='sum'
        ).fillna(0)
        
        # Convert to DataFrame for easier handling
        monthly_stats = pd.DataFrame(index=monthly_pivot.index)
        
        # Add monthly total spending (expenses only)
        expenses_cols = [col for col in monthly_pivot.columns if col != 'Income']
        monthly_stats['total_expenses'] = monthly_pivot[expenses_cols].sum(axis=1)
        
        if 'Income' in monthly_pivot.columns:
            monthly_stats['total_income'] = monthly_pivot['Income']
            monthly_stats['savings_rate'] = 1 - (monthly_stats['total_expenses'] / monthly_stats['total_income'])
        
        # Calculate 3-month rolling averages for main spending categories
        for category in monthly_pivot.columns:
            col_name = f"{category.lower()}_3m_avg"
            monthly_stats[col_name] = monthly_pivot[category].rolling(window=3, min_periods=1).mean()
        
        # Merge the monthly stats back to the original DataFrame
        df['month_year_key'] = df['month_year']  # Create a joining key
        monthly_stats_reset = monthly_stats.reset_index()
        
        # For each transaction, add the monthly stats
        result = pd.merge(
            df, 
            monthly_stats_reset, 
            left_on='month_year_key', 
            right_on='month_year',
            how='left'
        )
        
        # Clean up temporary columns
        result = result.drop(['month_year_key', 'month_year_y'], axis=1)
        result = result.rename(columns={'month_year_x': 'month_year'})
        
        # Calculate velocity features for individual transactions
        df_sorted = result.sort_values(['category', 'date'])
        
        # Calculate days since last transaction in same category
        result['days_since_last_same_category'] = df_sorted.groupby('category')['date'].diff().dt.days
        
        # Calculate average transaction amount in category
        avg_by_category = df_sorted.groupby('category')['amount_abs'].transform('mean')
        result['amount_vs_category_avg'] = result['amount_abs'] / avg_by_category
        
        # Mark outliers (transactions > 2x the category average)
        result['is_unusual'] = (result['amount_vs_category_avg'] > 2).astype(int)
        
        return result
    
    def prepare_for_embedding(self, df):
        """Create rich transaction descriptions for embedding"""
        df = df.copy()
        
        # Create a rich text description that combines relevant fields
        df['rich_description'] = df.apply(lambda row: self._create_rich_description(row), axis=1)
        
        return df
    
    def _create_rich_description(self, row):
        """Create a rich description for a transaction"""
        parts = []
        
        # Add merchant/description
        if pd.notnull(row.get('merchant')) and row.get('merchant') != "Unknown":
            parts.append(f"Merchant: {row['merchant']}")
        else:
            parts.append(f"Description: {row['description']}")
        
        # Add amount
        parts.append(f"Amount: ${abs(row['amount']):.2f}")
        
        # Add transaction type (expense/income)
        if row['amount'] < 0:
            parts.append("Type: Expense")
        else:
            parts.append("Type: Income")
        
        # Add category
        if pd.notnull(row.get('category')):
            parts.append(f"Category: {row['category']}")
        
        # Add account
        if pd.notnull(row.get('account')):
            parts.append(f"Account: {row['account']}")
        
        # Add date
        parts.append(f"Date: {row['date_iso']}")
        
        # Add recurring flag if present and true
        if row.get('is_recurring', 0) == 1:
            parts.append("Recurring: Yes")
        
        # Add transfer flag if present and true
        if row.get('is_transfer', 0) == 1:
            parts.append("Transfer: Yes")
        
        return " | ".join(parts)
    
    def standardize_all(self, df):
        """Apply all standardization steps"""
        # Date normalization
        df = self.standardize_date(df)
        
        # Amount standardization
        df = self.standardize_amount(df)
        
        # Category mapping
        df = self.map_categories(df)
        
        # Identify recurring transactions
        df = self.identify_recurring_transactions(df)
        
        # Identify transfers
        df = self.identify_transfers(df)
        
        # Calculate aggregated features
        df = self.calculate_aggregated_features(df)
        
        # Prepare for embedding
        df = self.prepare_for_embedding(df)
        
        return df 