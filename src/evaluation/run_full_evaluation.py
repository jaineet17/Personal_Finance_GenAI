#!/usr/bin/env python
"""
Script for running the complete Phase 5 evaluation pipeline:
1. Evaluation of system performance
2. Performance analysis to identify bottlenecks
3. System optimization based on analysis results
4. Re-evaluation after optimizations
5. Dashboard generation to visualize results
"""
import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.performance_analyzer import PerformanceAnalyzer
from src.evaluation.system_optimizer import SystemOptimizer
from src.evaluation.dashboard import load_results, create_html_dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("evaluation_pipeline")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run the complete evaluation pipeline for Finance RAG system")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save all evaluation results"
    )

    parser.add_argument(
        "--query-sets",
        type=str,
        default="basic_queries,intermediate_queries",
        help="Comma-separated list of query sets to evaluate or 'all'"
    )

    parser.add_argument(
        "--llm-provider",
        type=str,
        default="ollama",
        help="LLM provider to use"
    )

    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama3:latest",
        help="LLM model to use"
    )

    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Skip the optimization step"
    )

    parser.add_argument(
        "--skip-reevaluation",
        action="store_true",
        help="Skip the re-evaluation step after optimization"
    )

    parser.add_argument(
        "--analysis-queries",
        type=str,
        default=None,
        help="Comma-separated list of queries for performance analysis"
    )

    return parser.parse_args()


def run_initial_evaluation(args):
    """Run initial evaluation of system performance

    Args:
        args: Command line arguments

    Returns:
        str: Path to the evaluation results file
    """
    logger.info("Running initial evaluation")

    # Create a timestamp for the evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run the evaluation
    evaluator = RAGEvaluator(
        rag_system=None,  # Create a new RAG system with default parameters
        test_queries_path="tests/test_queries.json"
    )

    # Determine which query sets to evaluate
    if args.query_sets.lower() == "all":
        logger.info("Evaluating all query sets")
        evaluator.evaluate_all_query_sets()
    else:
        query_sets = [qs.strip() for qs in args.query_sets.split(",")]
        logger.info(f"Evaluating query sets: {', '.join(query_sets)}")

        for query_set in query_sets:
            evaluator.evaluate_query_set(query_set)

    # Save the results
    result_path = os.path.join(args.output_dir, f"evaluation_results_{timestamp}.json")
    evaluator.save_results(result_path)

    logger.info(f"Initial evaluation complete, results saved to {result_path}")

    return result_path, evaluator.rag_system


def run_performance_analysis(args, rag_system):
    """Run performance analysis

    Args:
        args: Command line arguments
        rag_system: RAG system to analyze

    Returns:
        str: Path to the analysis results file
    """
    logger.info("Running performance analysis")

    # Create a timestamp for the analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize the analyzer with the same RAG system used for evaluation
    analyzer = PerformanceAnalyzer(rag_system=rag_system)

    # Determine which queries to analyze
    if args.analysis_queries:
        queries = [q.strip() for q in args.analysis_queries.split(",")]
    else:
        # Default queries for analysis
        queries = [
            "How much did I spend on groceries in January 2025?",
            "What were my transportation expenses in March 2025?",
            "Compare my Instacart spending between February and March 2025"
        ]

    logger.info(f"Analyzing performance for {len(queries)} queries")

    # Run analysis for each query
    all_results = {}
    for query in queries:
        all_results[query] = analyzer.run_full_analysis(query)

    # Save results
    analysis_path = os.path.join(args.output_dir, f"performance_analysis_{timestamp}.json")
    analyzer.save_results(all_results, analysis_path)

    logger.info(f"Performance analysis complete, results saved to {analysis_path}")

    return analysis_path


def run_system_optimization(args, rag_system):
    """Run system optimization

    Args:
        args: Command line arguments
        rag_system: RAG system to optimize

    Returns:
        str: Path to the optimization results file
        object: Optimized RAG system
    """
    logger.info("Running system optimization")

    # Create a timestamp for the optimization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize the optimizer with the same RAG system used for evaluation and analysis
    optimizer = SystemOptimizer(rag_system=rag_system)

    # Apply all optimizations
    results = optimizer.apply_all_optimizations()

    # Save results
    optimization_path = os.path.join(args.output_dir, f"optimization_results_{timestamp}.json")
    optimizer.save_optimization_results(results, optimization_path)

    logger.info(f"System optimization complete, results saved to {optimization_path}")

    # Return the optimized RAG system
    return optimization_path, optimizer.rag_system


def run_post_optimization_evaluation(args, optimized_rag_system):
    """Run evaluation after optimization

    Args:
        args: Command line arguments
        optimized_rag_system: Optimized RAG system

    Returns:
        str: Path to the evaluation results file
    """
    logger.info("Running evaluation after optimization")

    # Create a timestamp for the evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run the evaluation with the optimized system
    evaluator = RAGEvaluator(
        rag_system=optimized_rag_system,
        test_queries_path="tests/test_queries.json"
    )

    # Use the same query sets as the initial evaluation
    if args.query_sets.lower() == "all":
        logger.info("Evaluating all query sets")
        evaluator.evaluate_all_query_sets()
    else:
        query_sets = [qs.strip() for qs in args.query_sets.split(",")]
        logger.info(f"Evaluating query sets: {', '.join(query_sets)}")

        for query_set in query_sets:
            evaluator.evaluate_query_set(query_set)

    # Save the results
    result_path = os.path.join(args.output_dir, f"evaluation_results_after_optimization_{timestamp}.json")
    evaluator.save_results(result_path)

    logger.info(f"Post-optimization evaluation complete, results saved to {result_path}")

    return result_path


def generate_dashboard(args):
    """Generate a dashboard from evaluation results

    Args:
        args: Command line arguments

    Returns:
        str: Path to the dashboard file
    """
    logger.info("Generating evaluation dashboard")

    # Load all evaluation results
    results = load_results(args.output_dir)

    if not results:
        logger.warning("No evaluation results found, skipping dashboard generation")
        return None

    # Create dashboard
    dashboard_path = create_html_dashboard(results, args.output_dir)

    logger.info(f"Dashboard generation complete, saved to {dashboard_path}")

    return dashboard_path


def main():
    """Main entry point for the evaluation pipeline"""
    # Parse command line arguments
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Starting complete evaluation pipeline")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Query sets: {args.query_sets}")
    logger.info(f"LLM provider: {args.llm_provider}")
    logger.info(f"LLM model: {args.llm_model}")

    # Step 1: Initial evaluation
    evaluation_path, rag_system = run_initial_evaluation(args)

    # Step 2: Performance analysis
    analysis_path = run_performance_analysis(args, rag_system)

    # Step 3: System optimization (optional)
    if args.skip_optimization:
        logger.info("Skipping optimization step")
        optimized_rag_system = rag_system
    else:
        optimization_path, optimized_rag_system = run_system_optimization(args, rag_system)

    # Step 4: Post-optimization evaluation (optional)
    if args.skip_optimization or args.skip_reevaluation:
        logger.info("Skipping post-optimization evaluation")
    else:
        post_evaluation_path = run_post_optimization_evaluation(args, optimized_rag_system)

    # Step 5: Generate dashboard
    dashboard_path = generate_dashboard(args)

    logger.info("Evaluation pipeline complete")

    # Print summary
    print("\n=== Evaluation Pipeline Summary ===")
    print(f"Initial evaluation results: {evaluation_path}")
    print(f"Performance analysis results: {analysis_path}")
    if not args.skip_optimization:
        print(f"Optimization results: {optimization_path}")
    if not (args.skip_optimization or args.skip_reevaluation):
        print(f"Post-optimization evaluation results: {post_evaluation_path}")
    if dashboard_path:
        print(f"Dashboard: {dashboard_path}")
    print("================================\n")

    return 0


if __name__ == "__main__":
    sys.exit(main()) 