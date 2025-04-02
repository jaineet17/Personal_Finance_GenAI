import os
import re
import pandas as pd
import hashlib
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
import uuid

class DataSourceImporter(ABC):
    """Base class for all data source importers"""
    
    @abstractmethod
    def import_data(self, file_path):
        """Import data from a specific source format"""
        pass
    
    @abstractmethod
    def anonymize_data(self, df):
        """Anonymize sensitive data"""
        pass
    
    @abstractmethod
    def clean_description(self, description):
        """Clean transaction description"""
        pass
    
    def anonymize_merchant(self, merchant_name):
        """Create a consistent pseudonym for a merchant"""
        if not merchant_name or pd.isna(merchant_name):
            return "Unknown Merchant"
            
        # Create a deterministic hash for the merchant name
        hash_obj = hashlib.md5(merchant_name.strip().lower().encode())
        # Use first 8 characters of hash as a suffix
        hash_suffix = hash_obj.hexdigest()[:8]
        
        # Extract first word or first letter of merchant name
        first_word = re.match(r'^([A-Za-z0-9]+)', merchant_name.strip())
        prefix = first_word.group(1) if first_word else "M"
        
        return f"{prefix}_{hash_suffix}"
    
    def scale_amount(self, amount, scale_factor=None):
        """Scale transaction amount for privacy"""
        if pd.isna(amount):
            return 0.0
            
        if scale_factor is None:
            # Generate a random scaling factor between 0.9 and 1.1
            # But use a fixed seed based on amount to keep it consistent
            seed = int(abs(float(amount)) * 100) % 1000
            np.random.seed(seed)
            scale_factor = np.random.uniform(0.9, 1.1)
            
        return round(float(amount) * scale_factor, 2)
    
    def create_account_id(self, account_info):
        """Create an anonymous account identifier"""
        if not account_info or pd.isna(account_info):
            return "unknown_account"
            
        # Create a deterministic hash for the account info
        hash_obj = hashlib.md5(str(account_info).strip().encode())
        # Use first 12 characters of hash
        hash_id = hash_obj.hexdigest()[:12]
        
        return f"acct_{hash_id}"


class ChaseCardImporter(DataSourceImporter):
    """Importer for Chase credit card data"""
    
    def __init__(self, account_name=None):
        self.account_name = account_name or "Chase Credit Card"
    
    def import_data(self, file_path):
        """Import Chase credit card data"""
        df = pd.read_csv(file_path)
        
        # Standardize column names
        columns = {
            'Transaction Date': 'date',
            'Post Date': 'post_date',
            'Description': 'description',
            'Category': 'original_category',
            'Type': 'transaction_type',
            'Amount': 'amount',
            'Memo': 'notes'
        }
        
        # Map only columns that exist
        valid_columns = {k: v for k, v in columns.items() if k in df.columns}
        df = df.rename(columns=valid_columns)
        
        # Add account column
        df['account'] = self.account_name
        
        # Ensure required columns exist
        required_columns = ['date', 'description', 'amount']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in the CSV file")
        
        return df
    
    def anonymize_data(self, df):
        """Anonymize Chase credit card data"""
        # Create a copy to avoid modifying the original dataframe
        df_anon = df.copy()
        
        # Anonymize merchant names in descriptions
        df_anon['merchant'] = df_anon['description'].apply(self.extract_merchant)
        df_anon['merchant_anon'] = df_anon['merchant'].apply(self.anonymize_merchant)
        
        # Scale transaction amounts
        df_anon['amount_original'] = df_anon['amount']
        df_anon['amount'] = df_anon['amount'].apply(self.scale_amount)
        
        # Create anonymous account ID
        df_anon['account_id'] = self.create_account_id(self.account_name)
        
        # Clean descriptions
        df_anon['description_original'] = df_anon['description']
        df_anon['description'] = df_anon['description'].apply(self.clean_description)
        
        return df_anon
    
    def extract_merchant(self, description):
        """Extract merchant name from description"""
        if pd.isna(description):
            return "Unknown"
            
        # Remove common prefixes
        prefixes = [
            "POS PURCHASE", "DEBIT PURCHASE", "SQ *", "TST*", "PAYPAL *", 
            "APPLE.COM/BILL", "ACH PAYMENT", "PAYMENT", "TRANSFER"
        ]
        
        clean_desc = description
        for prefix in prefixes:
            if description.startswith(prefix):
                clean_desc = description[len(prefix):].strip()
                break
        
        # Extract first part of description (likely the merchant name)
        merchant_match = re.search(r'^([^0-9*#]+)', clean_desc)
        if merchant_match:
            return merchant_match.group(1).strip()
        
        return clean_desc.split()[0] if clean_desc else "Unknown"
    
    def clean_description(self, description):
        """Clean transaction description"""
        if pd.isna(description):
            return "Unknown Transaction"
            
        # Replace multiple spaces with a single space
        clean_desc = re.sub(r'\s+', ' ', description.strip())
        
        # Remove common prefixes
        prefixes = [
            "POS PURCHASE", "DEBIT PURCHASE", "SQ *", "TST*", "PAYPAL *", 
            "APPLE.COM/BILL", "ACH PAYMENT"
        ]
        
        for prefix in prefixes:
            if clean_desc.startswith(prefix):
                clean_desc = clean_desc[len(prefix):].strip()
                break
        
        # Remove trailing numbers and special characters
        clean_desc = re.sub(r'[^A-Za-z0-9\s].*$', '', clean_desc).strip()
        
        return clean_desc


class ChaseAccountImporter(DataSourceImporter):
    """Importer for Chase bank account data"""
    
    def __init__(self, account_name=None):
        self.account_name = account_name or "Chase Bank Account"
    
    def import_data(self, file_path):
        """Import Chase bank account data"""
        df = pd.read_csv(file_path)
        
        # Standardize column names
        columns = {
            'Details': 'transaction_type',
            'Posting Date': 'date',
            'Description': 'description',
            'Amount': 'amount',
            'Type': 'payment_type',
            'Balance': 'balance',
            'Check or Slip #': 'check_number'
        }
        
        # Map only columns that exist
        valid_columns = {k: v for k, v in columns.items() if k in df.columns}
        df = df.rename(columns=valid_columns)
        
        # Add account column
        df['account'] = self.account_name
        
        # Ensure required columns exist
        required_columns = ['date', 'description', 'amount']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in the CSV file")
        
        # For chase bank account, DEBIT is negative and CREDIT is positive
        if 'transaction_type' in df.columns:
            df.loc[df['transaction_type'] == 'DEBIT', 'amount'] = -abs(df['amount'])
        
        return df
    
    def anonymize_data(self, df):
        """Anonymize Chase bank account data"""
        # Create a copy to avoid modifying the original dataframe
        df_anon = df.copy()
        
        # Anonymize merchant names in descriptions
        df_anon['merchant'] = df_anon['description'].apply(self.extract_merchant)
        df_anon['merchant_anon'] = df_anon['merchant'].apply(self.anonymize_merchant)
        
        # Scale transaction amounts
        df_anon['amount_original'] = df_anon['amount']
        df_anon['amount'] = df_anon['amount'].apply(self.scale_amount)
        
        # Create anonymous account ID
        df_anon['account_id'] = self.create_account_id(self.account_name)
        
        # Clean descriptions
        df_anon['description_original'] = df_anon['description']
        df_anon['description'] = df_anon['description'].apply(self.clean_description)
        
        return df_anon
    
    def extract_merchant(self, description):
        """Extract merchant name from description"""
        if pd.isna(description):
            return "Unknown"
            
        # Remove quotes if present
        if description.startswith('"') and description.endswith('"'):
            description = description[1:-1]
        
        # Extract the first part until special characters
        merchant_match = re.search(r'^([^0-9*#]+)', description)
        if merchant_match:
            return merchant_match.group(1).strip()
        
        return description.split()[0] if description else "Unknown"
    
    def clean_description(self, description):
        """Clean transaction description"""
        if pd.isna(description):
            return "Unknown Transaction"
            
        # Remove quotes if present
        if isinstance(description, str) and description.startswith('"') and description.endswith('"'):
            description = description[1:-1]
        
        # Replace multiple spaces with a single space
        clean_desc = re.sub(r'\s+', ' ', str(description).strip())
        
        # Remove web IDs and similar patterns
        clean_desc = re.sub(r'WEB ID: \d+', '', clean_desc)
        clean_desc = re.sub(r'PPD ID: \d+', '', clean_desc)
        
        # Replace common patterns
        clean_desc = re.sub(r'ACH (DEBIT|CREDIT)', '', clean_desc)
        
        return clean_desc.strip()


class GenericBankImporter(DataSourceImporter):
    """Importer for generic bank/credit card data"""
    
    def __init__(self, account_name=None):
        self.account_name = account_name or "Generic Account"
    
    def import_data(self, file_path):
        """Import data from a generic CSV format"""
        df = pd.read_csv(file_path)
        
        # Standardize column names - adapt based on your actual CSV format
        columns = {
            'Date': 'date',
            'Description': 'description',
            'Amount': 'amount',
            'Category': 'original_category',
            'Extended Details': 'notes',
            'Reference': 'reference',
        }
        
        # Map only columns that exist
        valid_columns = {k: v for k, v in columns.items() if k in df.columns}
        df = df.rename(columns=valid_columns)
        
        # Add account column
        df['account'] = self.account_name
        
        # Ensure required columns exist
        required_columns = ['date', 'description', 'amount']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in the CSV file")
        
        return df
    
    def anonymize_data(self, df):
        """Anonymize generic bank data"""
        # Create a copy to avoid modifying the original dataframe
        df_anon = df.copy()
        
        # Anonymize merchant names in descriptions
        df_anon['merchant'] = df_anon['description'].apply(self.extract_merchant)
        df_anon['merchant_anon'] = df_anon['merchant'].apply(self.anonymize_merchant)
        
        # Scale transaction amounts
        df_anon['amount_original'] = df_anon['amount']
        df_anon['amount'] = df_anon['amount'].apply(self.scale_amount)
        
        # Create anonymous account ID
        df_anon['account_id'] = self.create_account_id(self.account_name)
        
        # Clean descriptions
        df_anon['description_original'] = df_anon['description']
        df_anon['description'] = df_anon['description'].apply(self.clean_description)
        
        return df_anon
    
    def extract_merchant(self, description):
        """Extract merchant name from description"""
        if pd.isna(description):
            return "Unknown"
            
        # Extract first part (likely the merchant name)
        parts = str(description).split(',')
        if parts:
            return parts[0].strip()
        
        return "Unknown"
    
    def clean_description(self, description):
        """Clean transaction description"""
        if pd.isna(description):
            return "Unknown Transaction"
            
        # Replace multiple spaces with a single space
        clean_desc = re.sub(r'\s+', ' ', str(description).strip())
        
        # Take only the first part of the description (before commas)
        parts = clean_desc.split(',')
        if parts:
            return parts[0].strip()
        
        return clean_desc


# Example usage
if __name__ == "__main__":
    # Create an instance of ChaseCardImporter
    chase_card_importer = ChaseCardImporter()
    
    # Import data
    file_path = "path_to_your_chase_card_data.csv"
    df = chase_card_importer.import_data(file_path)
    
    # Anonymize data
    df_anon = chase_card_importer.anonymize_data(df)
    
    # Print anonymized data
    print(df_anon) 