# Finance RAG System Evaluation Framework

This directory contains the evaluation framework for the Finance RAG system, implemented as part of Phase 5. The framework allows for comprehensive performance assessment, bottleneck identification, system optimization, and visualization of results.

## Components

### 1. Metrics (`metrics.py`)
- Defines performance metrics for evaluating the RAG system
- Includes `ResponseMetrics`, `PerformanceTimer`, and `EvaluationResult` classes
- Captures response times, relevance scores, and memory usage

### 2. Evaluator (`evaluator.py`)
- Runs the RAG system against test queries from `tests/test_queries.json`
- Collects performance metrics for each query
- Calculates relevance scores based on expected properties
- Saves results to JSON files

### 3. Performance Analyzer (`performance_analyzer.py`)
- Performs detailed system profiling
- Measures component-level timings:
  - Retrieval time
  - Context assembly time
  - LLM inference time
- Tracks memory usage
- Identifies system bottlenecks

### 4. System Optimizer (`system_optimizer.py`)
- Implements optimizations for system components:
  - Retrieval optimizations (result count, query caching)
  - Context assembly optimizations (summarization, filtering)
  - LLM optimizations (temperature, response caching)
- Records applied optimizations and their effects

### 5. Dashboard (`dashboard.py`)
- Creates visualizations of evaluation results
- Generates HTML dashboard with charts
- Shows performance trends over time
- Provides component-level breakdown

### 6. Pipeline Scripts
- `run_evaluation.py`: Script to run just the evaluation
- `run_full_evaluation.py`: Script to run the complete pipeline

## Usage

### Basic Evaluation

To run a basic evaluation:

```bash
# From the project root
./run_evaluation.sh --query-sets basic_queries
```

This will:
1. Evaluate the queries in the specified query set
2. Save results to `evaluation_results/`
3. Generate a summary

### Full Evaluation Pipeline

To run the complete evaluation and optimization pipeline:

```bash
# From the project root
./run_optimization_pipeline.sh
```

This will:
1. Run initial evaluation
2. Perform performance analysis
3. Apply system optimizations
4. Re-evaluate after optimization
5. Generate a dashboard

### Customization

The evaluation framework can be customized in several ways:

1. **Test Queries**: Add new test queries to `tests/test_queries.json`
2. **Metrics**: Extend `ResponseMetrics` in `metrics.py` to capture additional metrics
3. **Optimizations**: Add new optimization techniques to `system_optimizer.py`
4. **Visualizations**: Add new charts and visualizations to `dashboard.py`

## Documentation

For more detailed documentation, see:

- `docs/EVALUATION.md`: Comprehensive documentation of the evaluation framework
- `docs/implementation_plan_phases_4-6.md`: Overall implementation plan including Phase 5 