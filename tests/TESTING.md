# Testing the Finance LLM Application

This document provides instructions on how to test the Finance LLM application using Ollama as the LLM provider.

## Prerequisites

Before running the tests, make sure you have:

1. **Ollama** installed on your system
   - Install from [ollama.com](https://ollama.com/)
   - Ensure the `ollama` command is available in your PATH

2. **llama3 model** pulled in Ollama
   - The test scripts will attempt to pull this if it's not already available
   - You can manually pull it with: `ollama pull llama3`

3. **Python dependencies** installed
   - Install with: `pip install -r requirements.txt`
   - Testing dependencies: `pip install pytest pytest-cov html-testRunner`

## Running Tests

### Method 1: Using the Test Script (Recommended)

We provide a convenient script to run all tests with Ollama properly configured:

```bash
./run_tests_with_ollama.sh
```

This script will:
- Check if Ollama is installed and running
- Start Ollama if it's not already running
- Pull the llama3 model if needed
- Configure the environment for testing
- Run the tests
- Export test results to the `test_output` directory

#### Script Options

The script supports several options:

```bash
./run_tests_with_ollama.sh [OPTIONS]

Options:
  -m, --module MODULE    Run specific test module
  -r, --report           Generate HTML test report
  -l, --list             List available test modules
  -h, --help             Show this help message
```

Examples:

```bash
# Run only the RAG system tests
./run_tests_with_ollama.sh -m test_rag_system

# Generate an HTML test report
./run_tests_with_ollama.sh -r

# List available test modules
./run_tests_with_ollama.sh -l
```

### Method 2: Manual Testing

If you prefer to manually run the tests:

1. Start Ollama:
   ```bash
   ollama serve
   ```

2. Set environment variables:
   ```bash
   export OLLAMA_API_BASE="http://localhost:11434"
   export ENVIRONMENT="test"
   ```

3. Run the tests:
   ```bash
   python -m tests.run_tests
   ```

## Test Output

Test results are saved to the `test_output` directory in JSON format, with filenames incorporating timestamps:

```
test_output/test_results_YYYYMMDD_HHMMSS.json
```

If you generate HTML reports, they will be available in the `test_reports` directory.

## Troubleshooting

### Ollama Connection Issues

If tests fail due to Ollama connection issues:

1. Verify Ollama is running:
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. Check if the llama3 model is available:
   ```bash
   ollama list
   ```

3. Try running with the environment explicitly set to "test" for mock responses:
   ```bash
   ENVIRONMENT=test python -m tests.run_tests
   ```

### Test Failures

If you encounter test failures:

1. Check the test output for specific errors
2. Verify that the llama3 model is properly loaded in Ollama
3. Make sure you have the required Python dependencies installed
4. Try running individual test modules to isolate issues 