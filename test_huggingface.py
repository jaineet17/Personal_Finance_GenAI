#!/usr/bin/env python

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.llm.llm_client import LLMClient
from src.rag.finance_rag import FinanceRAG

def main():
    # Print environment variables
    print(f"Environment variables:")
    print(f"DEFAULT_LLM_PROVIDER: {os.environ.get('DEFAULT_LLM_PROVIDER', 'Not set')}")
    print(f"DEFAULT_LLM_MODEL: {os.environ.get('DEFAULT_LLM_MODEL', 'Not set')}")
    print(f"HUGGINGFACE_API_KEY: {'Set' if os.environ.get('HUGGINGFACE_API_KEY') else 'Not set'}")
    
    # Initialize LLM client directly with explicit provider
    llm = LLMClient(
        provider="huggingface",
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        api_key=os.environ.get("HUGGINGFACE_API_KEY")
    )
    print(f"\nInitialized LLM client with provider: {llm.provider}")
    print(f"Using model: {llm.model_name}")
    
    # Initialize FinanceRAG with explicit provider
    rag = FinanceRAG(
        llm_provider="huggingface",
        llm_model="mistralai/Mistral-7B-Instruct-v0.2"
    )
    print(f"Initialized FinanceRAG with provider: {rag.llm.provider}")
    print(f"Using model: {rag.llm.model_name}")
    
    # Test with a simple query
    query = "What is the capital of France?"
    print(f"\nTesting with query: '{query}'")
    response = llm.generate_text(query)
    print(f"Response from {llm.provider} model: {response[:100]}...")

if __name__ == "__main__":
    main() 