import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_processing.init_db import create_database, export_category_mapping_json
from src.data_processing.transaction_processor import TransactionProcessor
from src.data_processing.unified_processor import UnifiedProcessor
from src.llm.ollama_service import OllamaService
from src.retrieval.rag_engine import RAGEngine

load_dotenv()

def init_app():
    """Initialize the application by creating necessary resources"""
    print("Initializing finance LLM application...")
    
    # Create database
    create_database()
    
    # Export category mapping
    export_category_mapping_json()
    
    # Check Ollama connection
    ollama = OllamaService()
    if ollama.check_availability():
        print("✓ Connected to Ollama server")
    else:
        print("✗ Could not connect to Ollama server at", os.getenv("OLLAMA_API_BASE"))
        print("  Make sure Ollama is running: ollama serve")
    
    print("Initialization complete!")

def process_data(file_path, **kwargs):
    """Process a CSV file of transactions"""
    processor = TransactionProcessor()
    
    print(f"Processing file: {file_path}")
    num_saved, embedding_success = processor.process_file(file_path, **kwargs)
    
    print(f"✓ {num_saved} transactions saved to database")
    if embedding_success:
        print("✓ Transactions successfully embedded in vector store")
    else:
        print("✗ Failed to embed transactions in vector store")

def process_all_data(source_type=None, account_name=None):
    """Process all data files with enhanced functionality"""
    processor = UnifiedProcessor()
    
    # If specific source_type and account_name were provided, create a configuration
    file_configs = None
    if source_type:
        # Find all CSV files in the raw data directory
        raw_data_path = Path(os.getenv("RAW_DATA_PATH", "./data/raw"))
        file_paths = list(raw_data_path.glob('*.csv'))
        
        file_configs = [
            {
                'file_path': str(path),
                'source_type': source_type,
                'account_name': account_name or path.stem
            }
            for path in file_paths
        ]
    
    # Process all files
    total, successful = processor.process_all_files(file_configs)
    
    print(f"✓ Processed {total} transactions from {successful} files")
    return total, successful

def process_accounts(accounts_config):
    """Process files for specific account configurations"""
    processor = UnifiedProcessor()
    
    file_configs = []
    for config in accounts_config:
        file_configs.append({
            'file_path': config['file_path'],
            'source_type': config['source_type'],
            'account_name': config['account_name']
        })
    
    # Process all files
    total, successful = processor.process_all_files(file_configs)
    
    print(f"✓ Processed {total} transactions from {successful} files")
    return total, successful

def chat_mode(model_name="llama3:8b"):
    """Interactive chat with the finance assistant"""
    rag_engine = RAGEngine(model_name=model_name)
    ollama = OllamaService(model_name=model_name)
    
    print(f"\nFinance Assistant (using {model_name})")
    print("=" * 50)
    print("Type your financial questions or 'exit' to quit.")
    
    # Test Ollama availability
    if not ollama.check_availability():
        print("Could not connect to Ollama server. Make sure it's running.")
        return
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        print("\nThinking...")
        response = rag_engine.generate_response(user_input)
        print("\nAssistant:", response)

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="Finance LLM Application")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Initialize command
    init_parser = subparsers.add_parser("init", help="Initialize the application")
    
    # Process data command
    process_parser = subparsers.add_parser("process", help="Process a CSV file of transactions")
    process_parser.add_argument("file", help="Path to the CSV file")
    process_parser.add_argument("--date-col", default="Date", help="Name of the date column")
    process_parser.add_argument("--desc-col", default="Description", help="Name of the description column")
    process_parser.add_argument("--amount-col", default="Amount", help="Name of the amount column")
    process_parser.add_argument("--category-col", help="Name of the category column")
    process_parser.add_argument("--account-col", help="Name of the account column")
    
    # Process all data command
    processall_parser = subparsers.add_parser("processall", help="Process all CSV files in the raw data directory")
    processall_parser.add_argument("--source-type", choices=["chase_card", "chase_account", "generic"], 
                                  help="Source type for all files")
    processall_parser.add_argument("--account-name", help="Account name for all files")
    
    # Process accounts command
    accounts_parser = subparsers.add_parser("accounts", help="Process data for defined accounts")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with the finance assistant")
    chat_parser.add_argument("--model", default="llama3:8b", help="Model to use for chat")
    
    args = parser.parse_args()
    
    if args.command == "init":
        init_app()
    elif args.command == "process":
        process_data(
            args.file,
            date_col=args.date_col,
            desc_col=args.desc_col,
            amount_col=args.amount_col,
            category_col=args.category_col,
            account_col=args.account_col
        )
    elif args.command == "processall":
        process_all_data(source_type=args.source_type, account_name=args.account_name)
    elif args.command == "accounts":
        # Define account configurations
        accounts_config = [
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
        process_accounts(accounts_config)
    elif args.command == "chat":
        chat_mode(model_name=args.model)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 