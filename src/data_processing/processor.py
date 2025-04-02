import os
import csv
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Processes financial transaction data from various sources.
    """
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the data processor.
        
        Args:
            input_dir: Directory containing raw data files
            output_dir: Directory to write processed data
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Initialized DataProcessor with input_dir={input_dir}, output_dir={output_dir}")
    
    def detect_file_type(self, file_path: str) -> str:
        """
        Detect the type of financial data file.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            String identifying the file type (csv, ofx, qfx, etc.)
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        # Try to determine type based on extension
        if file_ext == '.csv':
            return 'csv'
        elif file_ext == '.ofx':
            return 'ofx'
        elif file_ext == '.qfx':
            return 'qfx'
        elif file_ext == '.json':
            return 'json'
        elif file_ext == '.xlsx':
            return 'excel'
        
        # If extension is not recognized, try to detect based on content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                
                # Check for CSV-like content (comma-separated values)
                if ',' in first_line:
                    return 'csv'
                
                # Check for OFX/QFX content (XML-like)
                if '<OFX>' in first_line or '<OFX>' in f.read(1000):
                    return 'ofx'
                
                # Check for JSON content
                if first_line.startswith('{') or first_line.startswith('['):
                    return 'json'
        except Exception as e:
            logger.warning(f"Error detecting file type from content: {e}")
        
        # Default to unknown
        logger.warning(f"Unable to detect file type for {file_path}, defaulting to 'unknown'")
        return 'unknown'
    
    def process_files(self) -> List[Path]:
        """
        Process all CSV files in the input directory.
        
        Returns:
            List of paths to the processed output files
        """
        processed_files = []
        
        # Find all CSV files in the input directory
        csv_files = list(self.input_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        for csv_file in csv_files:
            output_file = self._process_file(csv_file)
            if output_file:
                processed_files.append(output_file)
        
        return processed_files
    
    def _process_file(self, file_path: Path) -> Optional[Path]:
        """
        Process a single file and standardize the data.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Path to the processed output file, or None if processing failed
        """
        try:
            # Read the CSV file
            logger.info(f"Processing file: {file_path}")
            df = pd.read_csv(file_path)
            
            # Basic validation - check if required columns exist
            required_columns = self._infer_required_columns(df)
            
            # Standardize column names
            df = self._standardize_columns(df, required_columns)
            
            # Standardize date format
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # Standardize amount format (ensure 2 decimal places)
            if 'amount' in df.columns:
                df['amount'] = df['amount'].round(2)
            
            # Add transaction ID if not present
            if 'id' not in df.columns:
                df['id'] = [f"tx_{i+1:06d}" for i in range(len(df))]
                
            # Write to output file
            output_file = self.output_dir / f"processed_{file_path.stem}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Wrote processed data to {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None
    
    def _infer_required_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Infer the mapping of required columns in the input data.
        
        Args:
            df: DataFrame containing the input data
            
        Returns:
            Dictionary mapping standard column names to input column names
        """
        columns = df.columns
        column_map = {}
        
        # Look for date column
        date_candidates = [col for col in columns if 'date' in col.lower()]
        if date_candidates:
            column_map['date'] = date_candidates[0]
        
        # Look for description column
        desc_candidates = [col for col in columns if col.lower() in ['description', 'desc', 'memo', 'transaction', 'narration']]
        if desc_candidates:
            column_map['description'] = desc_candidates[0]
        
        # Look for amount column
        amount_candidates = [col for col in columns if col.lower() in ['amount', 'sum', 'value', 'transaction_amount']]
        if amount_candidates:
            column_map['amount'] = amount_candidates[0]
        
        # Look for category column
        category_candidates = [col for col in columns if col.lower() in ['category', 'cat', 'type', 'transaction_type']]
        if category_candidates:
            column_map['category'] = category_candidates[0]
        
        # Look for account column
        account_candidates = [col for col in columns if col.lower() in ['account', 'acct', 'source']]
        if account_candidates:
            column_map['account'] = account_candidates[0]
        
        logger.info(f"Inferred column mapping: {column_map}")
        return column_map
    
    def _standardize_columns(self, df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
        """
        Standardize column names in the DataFrame.
        
        Args:
            df: DataFrame to standardize
            column_map: Mapping of standard column names to input column names
            
        Returns:
            DataFrame with standardized column names
        """
        # Create a new DataFrame with standard column names
        new_df = pd.DataFrame()
        
        # Copy columns based on the mapping
        for standard_name, input_name in column_map.items():
            new_df[standard_name] = df[input_name]
        
        # Copy any unmapped columns as-is
        for col in df.columns:
            if col not in column_map.values() and col not in new_df.columns:
                new_df[col] = df[col]
        
        return new_df 