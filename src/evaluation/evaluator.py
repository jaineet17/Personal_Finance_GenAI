"""
Evaluation framework for the Finance RAG system.
This module provides automated evaluation of the RAG system using existing test data.
"""
import os
import sys
import json
import time
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from src.evaluation.metrics import ResponseMetrics, PerformanceTimer, EvaluationResult
from src.rag.finance_rag import FinanceRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("rag_evaluator")


class RAGEvaluator:
    """Evaluator for the Finance RAG system"""

    def __init__(self, 
                 rag_system: Optional[FinanceRAG] = None,
                 test_queries_path: str = "tests/test_queries.json"):
        """Initialize the evaluator

        Args:
            rag_system (Optional[FinanceRAG]): RAG system to evaluate
            test_queries_path (str): Path to test queries JSON file
        """
        self.rag_system = rag_system or FinanceRAG(
            llm_provider="ollama",
            llm_model="llama3:latest",
            temperature=0.1  # Lower temperature for more deterministic responses
        )

        self.test_queries_path = test_queries_path
        self.test_queries = self._load_test_queries()
        self.evaluation_results = EvaluationResult()

    def _load_test_queries(self) -> Dict[str, List[Dict]]:
        """Load test queries from JSON file"""
        try:
            with open(self.test_queries_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading test queries: {e}")
            return {}

    def evaluate_query_set(self, query_set: str = "basic_queries") -> List[ResponseMetrics]:
        """Evaluate a set of queries

        Args:
            query_set (str): Name of query set from test_queries.json

        Returns:
            List[ResponseMetrics]: Metrics for each query
        """
        if query_set not in self.test_queries:
            logger.error(f"Query set {query_set} not found in test queries")
            return []

        queries = self.test_queries[query_set]
        results = []

        for i, query_data in enumerate(queries):
            logger.info(f"Evaluating query {i+1}/{len(queries)}: {query_data['query']}")

            # Process the query and collect metrics
            metrics = self.evaluate_single_query(query_data)
            results.append(metrics)

            # Add to evaluation results
            self.evaluation_results.add_response_metrics(metrics)

            # Add a small delay to avoid rate limiting
            time.sleep(0.5)

        return results

    def evaluate_single_query(self, query_data: Dict) -> ResponseMetrics:
        """Evaluate a single query

        Args:
            query_data (Dict): Query data from test_queries.json

        Returns:
            ResponseMetrics: Metrics for the query
        """
        query = query_data["query"]
        expected_properties = query_data.get("expected_properties", [])
        expected_retrieval_count = query_data.get("expected_retrieval_count", 0)

        # Measure overall response time
        total_timer = PerformanceTimer().start()

        # Measure retrieval time
        retrieval_timer = PerformanceTimer().start()
        # In a future implementation, we could hook into the retrieval system
        # to measure actual retrieval time and document count
        retrieval_time = retrieval_timer.stop()

        # Get memory usage before query
        memory_before = self._get_memory_usage()

        # Measure LLM time (approximation)
        llm_timer = PerformanceTimer().start()
        try:
            response = self.rag_system.query(query)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            response = f"Error: {str(e)}"
        llm_time = llm_timer.stop()

        # Get memory usage after query
        memory_after = self._get_memory_usage()
        memory_delta = memory_after - memory_before

        # Calculate total response time
        total_time = total_timer.stop()

        # Convert response to string if it's not already a string
        response_str = self._ensure_string_response(response)

        # Calculate metrics
        metrics = ResponseMetrics(
            query=query,
            response=response_str,
            response_time_ms=total_time,
            retrieval_time_ms=retrieval_time,
            llm_time_ms=llm_time,
            # Since we don't have exact values for these metrics,
            # we provide estimates to demonstrate the concept
            tokens_used=self._estimate_token_count(query, response_str),
            num_docs_retrieved=expected_retrieval_count,
        )

        # Calculate relevance score based on expected properties
        metrics.relevance_score = self._calculate_relevance_score(
            response_str, expected_properties
        )

        return metrics

    def evaluate_all_query_sets(self) -> Dict[str, List[ResponseMetrics]]:
        """Evaluate all query sets

        Returns:
            Dict[str, List[ResponseMetrics]]: Metrics for each query set
        """
        results = {}

        # Get the query set names, excluding test_transactions and categorization_examples
        query_sets = [
            name for name in self.test_queries.keys() 
            if name not in ["test_transactions", "categorization_examples"]
        ]

        # Evaluate each query set
        for query_set in query_sets:
            logger.info(f"Evaluating query set: {query_set}")
            results[query_set] = self.evaluate_query_set(query_set)

        # Update memory usage metrics
        self.evaluation_results.set_memory_usage({
            "current": self._get_memory_usage(),
            "peak": self._get_peak_memory_usage()
        })

        return results

    def save_results(self, output_path: str = "evaluation_results.json") -> None:
        """Save evaluation results to a file

        Args:
            output_path (str): Path to save results to
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.evaluation_results.to_dict(), f, indent=2)
            logger.info(f"Evaluation results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")

    def _ensure_string_response(self, response: Any) -> str:
        """Ensure that the response is a string

        Args:
            response: The response from the RAG system

        Returns:
            str: String representation of the response
        """
        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            try:
                return json.dumps(response, indent=2)
            except:
                return str(response)
        else:
            return str(response)

    def _calculate_relevance_score(self, response: str, expected_properties: List[str]) -> float:
        """Calculate relevance score based on expected properties

        Args:
            response (str): The RAG system's response
            expected_properties (List[str]): Expected properties to be in the response

        Returns:
            float: Relevance score between 0 and 1
        """
        # Convert to lowercase for case-insensitive matching
        if not response or not expected_properties:
            return 0.0 if expected_properties else 1.0

        response_lower = response.lower()

        # Count how many expected properties are in the response
        property_matches = sum(
            1 for prop in expected_properties 
            if prop.lower() in response_lower
        )

        # Calculate score
        return property_matches / len(expected_properties)

    def _estimate_token_count(self, query: str, response: str) -> int:
        """Estimate token count for query and response

        Args:
            query (str): The user query
            response (str): The RAG system's response

        Returns:
            int: Estimated token count
        """
        # Simple estimation: ~4 characters per token
        return (len(query) + len(response)) // 4

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB

        Returns:
            float: Memory usage in MB
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB

    def _get_peak_memory_usage(self) -> float:
        """Get peak memory usage in MB

        Returns:
            float: Peak memory usage in MB
        """
        return psutil.Process(os.getpid()).memory_info().peak_wset / (1024 * 1024) if hasattr(psutil.Process().memory_info(), 'peak_wset') else 0.0


def run_evaluation():
    """Run the evaluation and save results"""
    logger.info("Starting RAG system evaluation")

    evaluator = RAGEvaluator()
    evaluator.evaluate_all_query_sets()

    # Save results to file
    results_path = "evaluation_results.json"
    evaluator.save_results(results_path)

    # Get result summary
    result_dict = evaluator.evaluation_results.to_dict()
    summary = result_dict["summary"]

    logger.info("Evaluation complete")
    logger.info(f"Average response time: {summary['avg_response_time_ms']:.2f} ms")
    logger.info(f"Average relevance score: {summary['avg_relevance']:.2f}")
    logger.info(f"Total queries evaluated: {summary['total_queries']}")

    return results_path


if __name__ == "__main__":
    results_file = run_evaluation()
    print(f"Evaluation results saved to {results_file}") 