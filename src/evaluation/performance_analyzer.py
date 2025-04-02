"""
Performance analyzer for the Finance RAG system.
Analyzes system performance and identifies bottlenecks.
"""
import os
import time
import json
import logging
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from src.evaluation.metrics import PerformanceTimer
from src.rag.finance_rag import FinanceRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("performance_analyzer")


class PerformanceAnalyzer:
    """Analyzer for system performance"""
    
    def __init__(self, rag_system: Optional[FinanceRAG] = None):
        """Initialize the performance analyzer
        
        Args:
            rag_system (Optional[FinanceRAG]): RAG system to analyze
        """
        self.rag_system = rag_system or FinanceRAG(
            llm_provider="ollama",
            llm_model="llama3:latest",
            temperature=0.1
        )
        self.results = {}
        
    def measure_end_to_end_latency(self, query: str, iterations: int = 5) -> Dict[str, Any]:
        """Measure end-to-end latency for a query
        
        Args:
            query (str): Query to test
            iterations (int): Number of iterations
            
        Returns:
            Dict[str, Any]: Timing results
        """
        logger.info(f"Measuring end-to-end latency for query: '{query}'")
        
        response_times = []
        
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations}")
            
            # Start timer
            timer = PerformanceTimer().start()
            
            # Process query
            self.rag_system.query(query)
            
            # Stop timer
            elapsed = timer.stop()
            response_times.append(elapsed)
            
            # Wait to avoid rate limiting
            time.sleep(1)
        
        # Calculate statistics
        results = {
            "query": query,
            "iterations": iterations,
            "mean_response_time_ms": np.mean(response_times),
            "median_response_time_ms": np.median(response_times),
            "min_response_time_ms": np.min(response_times),
            "max_response_time_ms": np.max(response_times),
            "std_dev_ms": np.std(response_times),
            "raw_times_ms": response_times
        }
        
        logger.info(f"Results: Mean {results['mean_response_time_ms']:.2f} ms, "
                   f"Median {results['median_response_time_ms']:.2f} ms")
        
        return results
    
    def analyze_memory_usage(self, query: str) -> Dict[str, Any]:
        """Analyze memory usage during query processing
        
        Args:
            query (str): Query to process
            
        Returns:
            Dict[str, Any]: Memory usage statistics
        """
        logger.info(f"Analyzing memory usage for query: '{query}'")
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Process query
        self.rag_system.query(query)
        
        # Get peak memory usage
        peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        results = {
            "query": query,
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "delta_mb": peak_memory - initial_memory
        }
        
        logger.info(f"Memory usage: Initial {initial_memory:.2f} MB, "
                   f"Peak {peak_memory:.2f} MB, Delta {results['delta_mb']:.2f} MB")
        
        return results
    
    def analyze_system_components(self, query: str) -> Dict[str, Any]:
        """Analyze performance of individual system components
        
        Args:
            query (str): Query to process
            
        Returns:
            Dict[str, Any]: Component timing results
        """
        logger.info(f"Analyzing system components for query: '{query}'")
        
        # Initialize timers
        total_timer = PerformanceTimer().start()
        
        # We're measuring these steps independently as they aren't directly
        # accessible in the FinanceRAG implementation
        
        # 1. Measure retrieval time by accessing the retrieval system directly
        retrieval_timer = PerformanceTimer().start()
        retrieval_results = self.rag_system.retrieval.retrieve_similar_transactions(
            query=query, n_results=30
        )
        retrieval_time = retrieval_timer.stop()
        
        # 2. Measure context assembly time (approximation)
        context_timer = PerformanceTimer().start()
        context = self.rag_system.get_relevant_context(query)
        context_time = context_timer.stop()
        
        # 3. Process full query to get total time
        query_timer = PerformanceTimer().start()
        self.rag_system.query(query)
        query_time = query_timer.stop()
        
        # Calculate LLM inference time (approximation)
        llm_time = query_time - (retrieval_time + context_time)
        if llm_time < 0:
            llm_time = query_time * 0.8  # Fallback approximation
        
        # Stop total timer
        total_time = total_timer.stop()
        
        # Calculate overhead
        overhead = total_time - query_time
        
        results = {
            "query": query,
            "total_time_ms": total_time,
            "query_processing_time_ms": query_time,
            "retrieval_time_ms": retrieval_time,
            "context_assembly_time_ms": context_time,
            "llm_inference_time_ms": llm_time,
            "overhead_ms": overhead,
            "component_breakdown": {
                "retrieval_pct": (retrieval_time / query_time) * 100 if query_time > 0 else 0,
                "context_pct": (context_time / query_time) * 100 if query_time > 0 else 0,
                "llm_pct": (llm_time / query_time) * 100 if query_time > 0 else 0
            }
        }
        
        logger.info(f"Component breakdown: "
                   f"Retrieval {results['component_breakdown']['retrieval_pct']:.1f}%, "
                   f"Context {results['component_breakdown']['context_pct']:.1f}%, "
                   f"LLM {results['component_breakdown']['llm_pct']:.1f}%")
        
        return results
    
    def run_full_analysis(self, query: str, iterations: int = 3) -> Dict[str, Any]:
        """Run full performance analysis
        
        Args:
            query (str): Query to analyze
            iterations (int): Number of iterations for latency measurement
            
        Returns:
            Dict[str, Any]: Complete analysis results
        """
        logger.info(f"Running full performance analysis for query: '{query}'")
        
        # Run analyses
        latency_results = self.measure_end_to_end_latency(query, iterations)
        memory_results = self.analyze_memory_usage(query)
        component_results = self.analyze_system_components(query)
        
        # Combine results
        results = {
            "query": query,
            "timestamp": time.time(),
            "latency": latency_results,
            "memory": memory_results,
            "components": component_results
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_file: str = "performance_analysis.json"):
        """Save analysis results to a file
        
        Args:
            results (Dict[str, Any]): Analysis results
            output_file (str): Output file path
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Analysis results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")


def main():
    """Run performance analysis with sample queries"""
    logger.info("Starting performance analysis")
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer()
    
    # Sample queries to analyze
    queries = [
        "How much did I spend on groceries in January 2025?",
        "What were my transportation expenses in March 2025?",
        "Compare my Instacart spending between February and March 2025"
    ]
    
    # Run analysis for each query
    all_results = {}
    for query in queries:
        all_results[query] = analyzer.run_full_analysis(query)
    
    # Save results
    analyzer.save_results(all_results, "performance_analysis.json")
    
    logger.info("Performance analysis complete")


if __name__ == "__main__":
    main() 