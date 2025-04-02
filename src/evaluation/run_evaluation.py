#!/usr/bin/env python
"""
Script to run the evaluation framework for the Finance RAG system.
"""
import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.evaluation.evaluator import RAGEvaluator, run_evaluation
from src.rag.finance_rag import FinanceRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("evaluation_script")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run evaluation for Finance RAG system")
    
    parser.add_argument(
        "--query-sets",
        type=str,
        default="all",
        help="Comma-separated list of query sets to evaluate or 'all'"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="ollama",
        help="LLM provider to use (default: ollama)"
    )
    
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama3:latest",
        help="LLM model to use (default: llama3:latest)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for LLM generation (default: 0.1)"
    )
    
    parser.add_argument(
        "--test-queries",
        type=str,
        default="tests/test_queries.json",
        help="Path to test queries JSON file"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the evaluation script"""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the RAG system
    logger.info(f"Initializing FinanceRAG with {args.llm_provider}/{args.llm_model}")
    rag_system = FinanceRAG(
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        temperature=args.temperature
    )
    
    # Initialize the evaluator
    logger.info(f"Initializing evaluator with test queries from {args.test_queries}")
    evaluator = RAGEvaluator(
        rag_system=rag_system,
        test_queries_path=args.test_queries
    )
    
    # Determine which query sets to evaluate
    if args.query_sets.lower() == "all":
        logger.info("Evaluating all query sets")
        results = evaluator.evaluate_all_query_sets()
    else:
        query_sets = [qs.strip() for qs in args.query_sets.split(",")]
        logger.info(f"Evaluating query sets: {', '.join(query_sets)}")
        
        results = {}
        for query_set in query_sets:
            if query_set in evaluator.test_queries:
                results[query_set] = evaluator.evaluate_query_set(query_set)
            else:
                logger.warning(f"Query set '{query_set}' not found in test queries, skipping")
    
    # Generate timestamp for result filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"evaluation_results_{timestamp}.json"
    result_path = os.path.join(args.output_dir, result_filename)
    
    # Save the results
    evaluator.save_results(result_path)
    
    # Print summary
    result_dict = evaluator.evaluation_results.to_dict()
    summary = result_dict["summary"]
    
    logger.info("Evaluation complete")
    logger.info(f"Average response time: {summary['avg_response_time_ms']:.2f} ms")
    logger.info(f"Average relevance score: {summary['avg_relevance']:.2f}")
    logger.info(f"Total queries evaluated: {summary['total_queries']}")
    logger.info(f"Results saved to {result_path}")
    
    # Create a simple summary file
    summary_path = os.path.join(args.output_dir, f"summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Finance RAG Evaluation Summary\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"LLM Provider: {args.llm_provider}\n")
        f.write(f"LLM Model: {args.llm_model}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Query Sets: {args.query_sets}\n\n")
        f.write(f"Total Queries: {summary['total_queries']}\n")
        f.write(f"Average Response Time: {summary['avg_response_time_ms']:.2f} ms\n")
        f.write(f"Average Relevance Score: {summary['avg_relevance']:.2f}\n")
    
    logger.info(f"Summary saved to {summary_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 