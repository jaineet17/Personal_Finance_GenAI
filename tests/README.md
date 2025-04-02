# Finance RAG Testing Framework

This directory contains comprehensive tests for the Finance RAG application, focusing on phases 1-3 of development.

## Test Organization

The test suite is organized into the following modules:

- `test_data_processing.py`: Tests for data processing and database components
- `test_embedding_system.py`: Tests for embedding generation and vector store components
- `test_rag_system.py`: Tests for RAG system components including query processing and retrieval
- `test_integration.py`: End-to-end integration tests for the complete system

## Running Tests

You can run the test suite using the `run_tests.py` script:

```bash
# Run all tests
python tests/run_tests.py

# Run specific test modules
python tests/run_tests.py -m test_data_processing test_embedding_system

# List available test modules
python tests/run_tests.py -l

# Generate HTML test report
python tests/run_tests.py -r
```

## Test Coverage

The test suite covers the following aspects:

### Data Processing Tests
- Data integrity verification
- Data type validation
- Transformation accuracy
- Database validation
- Aggregate calculations

### Embedding System Tests
- Embedding quality assessment
- Embedding consistency
- Vector similarity calculations
- ChromaDB vector store operations
- Retrieval effectiveness

### RAG System Tests
- Query processing
- Context preparation
- Transaction categorization
- Spending analysis
- Conversation history handling

### Integration Tests
- End-to-end query flow
- Performance benchmarking
- Retrieval accuracy
- Data pipeline integrity

## Test Data

The tests generate synthetic financial data for testing purposes. No real financial data is used or required.

## Adding New Tests

To add new tests:

1. Create a new test file named `test_your_feature.py`
2. Subclass `unittest.TestCase`
3. Implement test methods that start with `test_`
4. Add the test module to the appropriate section in this README

## Evaluation Methodology

Tests use the following evaluation criteria:

1. **Retrieval Quality**: Are the most relevant transactions being retrieved?
2. **Response Accuracy**: Are numerical calculations correct?
3. **Response Completeness**: Does the response fully answer the query?
4. **Performance**: Is the response time reasonable?

## Test Requirements

Additional dependencies for testing:
```
pytest>=7.3.1
pytest-cov>=4.1.0
html-testRunner>=1.2.1
```

Install with:
```bash
pip install pytest pytest-cov html-testRunner
``` 