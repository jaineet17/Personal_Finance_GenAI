# Finance RAG System Evaluation Framework

This document provides details about the evaluation and optimization framework implemented in Phase 5 of the Finance RAG system. The framework enables comprehensive performance assessment, bottleneck identification, system optimization, and visualization of results.

## Overview

The evaluation framework consists of the following components:

1. **Metrics Collection**: Captures response time, retrieval performance, memory usage, and relevance scores
2. **Performance Analysis**: Identifies bottlenecks in the system components
3. **System Optimization**: Implements improvements based on performance analysis
4. **Evaluation Dashboard**: Visualizes performance metrics and improvements

## Components

### 1. Evaluation Module

The evaluation module (`src/evaluation/evaluator.py`) runs the RAG system against test queries and collects performance metrics:

- Response time
- Retrieval time
- LLM inference time
- Relevance scores (based on expected properties in responses)
- Memory usage

Usage:
```python
from src.evaluation.evaluator import RAGEvaluator

# Initialize evaluator
evaluator = RAGEvaluator()

# Evaluate all query sets
evaluator.evaluate_all_query_sets()

# Or evaluate specific query set
evaluator.evaluate_query_set("basic_queries")

# Save results
evaluator.save_results("evaluation_results.json")
```

### 2. Performance Analyzer

The performance analyzer (`src/evaluation/performance_analyzer.py`) performs detailed profiling of the system:

- End-to-end latency measurements
- Memory usage tracking
- Component-level timing (retrieval, context assembly, LLM inference)
- System bottleneck identification

Usage:
```python
from src.evaluation.performance_analyzer import PerformanceAnalyzer

# Initialize analyzer
analyzer = PerformanceAnalyzer()

# Run full analysis on a query
results = analyzer.run_full_analysis(
    query="How much did I spend on groceries in January 2025?"
)

# Save results
analyzer.save_results(results, "performance_analysis.json")
```

### 3. System Optimizer

The system optimizer (`src/evaluation/system_optimizer.py`) implements optimizations to improve performance:

- Retrieval optimization (reduced result count, query caching)
- Context assembly optimization (transaction summarization, query-based filtering)
- LLM parameter optimization (temperature adjustment, response caching)

Usage:
```python
from src.evaluation.system_optimizer import SystemOptimizer

# Initialize optimizer
optimizer = SystemOptimizer()

# Apply all optimizations
results = optimizer.apply_all_optimizations()

# Or apply specific optimizations
retrieval_results = optimizer.optimize_retrieval()
context_results = optimizer.optimize_context_assembly()
llm_results = optimizer.optimize_llm_parameters()

# Save results
optimizer.save_optimization_results(results, "optimization_results.json")
```

### 4. Visualization Dashboard

The dashboard module (`src/evaluation/dashboard.py`) creates visual representations of evaluation results:

- Performance trends over time
- Query-specific performance metrics
- Response time breakdown by component
- Comparative analysis of before/after optimization

Usage:
```python
from src.evaluation.dashboard import load_results, create_html_dashboard

# Load evaluation results
results = load_results("evaluation_results")

# Create dashboard
dashboard_path = create_html_dashboard(results, "evaluation_results")
```

## Running the Evaluation Pipeline

For convenience, the entire evaluation and optimization pipeline can be run with a single command:

```bash
./run_optimization_pipeline.sh
```

This script:
1. Runs the initial evaluation
2. Performs performance analysis
3. Applies system optimizations
4. Re-evaluates after optimization
5. Generates a dashboard

### Command Line Options

The pipeline supports several command line options:

```bash
./run_optimization_pipeline.sh --query-sets all --skip-optimization
```

Available options:
- `--output-dir`: Directory to save results (default: evaluation_results)
- `--query-sets`: Comma-separated list of query sets to evaluate
- `--llm-provider`: LLM provider to use (default: ollama) 
- `--llm-model`: LLM model to use (default: llama3:latest)
- `--skip-optimization`: Skip the optimization step
- `--skip-reevaluation`: Skip the re-evaluation step after optimization
- `--analysis-queries`: Comma-separated list of queries for performance analysis

## Advanced Usage

For more control, you can run the evaluation components individually:

```bash
# Run just the evaluation
python src/evaluation/run_evaluation.py --query-sets basic_queries

# Run just performance analysis
python src/evaluation/performance_analyzer.py

# Run just system optimization
python src/evaluation/system_optimizer.py

# Generate a dashboard from existing results
python src/evaluation/dashboard.py --results-dir evaluation_results
```

## Interpreting Results

The evaluation framework produces several types of results:

1. **Evaluation Results**: JSON files containing metrics for each query
2. **Performance Analysis**: Detailed breakdown of system performance
3. **Optimization Results**: List of applied optimizations and their effects
4. **Dashboard**: HTML visualization of results

The dashboard provides the most user-friendly way to interpret results, showing:
- Response time trends
- Relevance score improvements
- Component-level performance breakdown
- Memory usage patterns

## Extending the Framework

The evaluation framework is designed to be extensible:

- Add new metrics to `ResponseMetrics` class in `metrics.py`
- Implement new optimizations in `system_optimizer.py`
- Add new visualization components to `dashboard.py`
- Create custom evaluation pipelines in `run_full_evaluation.py` 