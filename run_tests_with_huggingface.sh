#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Finance LLM App Test Runner (Hugging Face)${NC}"

# Check if Hugging Face API key is set
if [ -z "$HUGGINGFACE_API_KEY" ]; then
    # Try to load from .env file
    if [ -f .env ]; then
        source .env
    fi
    
    if [ -z "$HUGGINGFACE_API_KEY" ]; then
        echo -e "${RED}Error: HUGGINGFACE_API_KEY is not set. Please set it in your .env file or as an environment variable.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Found Hugging Face API key.${NC}"

# Create output directories
mkdir -p test_output

# Modify test files to use Hugging Face
echo "Temporarily modifying test files to use Hugging Face..."

# Replace Ollama with Hugging Face in test files
find tests/ -type f -name "*.py" -exec sed -i '' 's/llm_provider="ollama"/llm_provider="huggingface"/g' {} \;
find tests/ -type f -name "*.py" -exec sed -i '' 's/provider="ollama"/provider="huggingface"/g' {} \;
find tests/ -type f -name "*.py" -exec sed -i '' 's/llm_model="llama3"/llm_model="mistralai\/Mistral-7B-Instruct-v0.2"/g' {} \;
find tests/ -type f -name "*.py" -exec sed -i '' 's/model_name="llama3:latest"/model_name="mistralai\/Mistral-7B-Instruct-v0.2"/g' {} \;
find tests/ -type f -name "*.py" -exec sed -i '' 's/llm_model="llama3:latest"/llm_model="mistralai\/Mistral-7B-Instruct-v0.2"/g' {} \;

# Set environment variables for testing
export DEFAULT_LLM_PROVIDER="huggingface"
export DEFAULT_LLM_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
export ENVIRONMENT="test"

# Run the tests
echo -e "${GREEN}Running tests with Hugging Face...${NC}"
echo -e "${YELLOW}Test results will be saved to test_output/ directory${NC}"

# Parse command line arguments
TEST_MODULE=""
GENERATE_REPORT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--module)
            TEST_MODULE="$2"
            shift 2
            ;;
        -r|--report)
            GENERATE_REPORT=true
            shift
            ;;
        -l|--list)
            echo -e "${YELLOW}Available test modules:${NC}"
            python -m tests.run_tests -l
            exit 0
            ;;
        -h|--help)
            echo -e "Usage: $0 [OPTIONS]"
            echo -e "  -m, --module MODULE    Run specific test module"
            echo -e "  -r, --report           Generate HTML test report"
            echo -e "  -l, --list             List available test modules"
            echo -e "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Construct test command
TEST_CMD="python -m tests.run_tests"

if [ -n "$TEST_MODULE" ]; then
    TEST_CMD="$TEST_CMD -m $TEST_MODULE"
fi

if [ "$GENERATE_REPORT" = true ]; then
    TEST_CMD="$TEST_CMD -r"
fi

# Run the tests
echo -e "${YELLOW}Executing: $TEST_CMD${NC}"
$TEST_CMD
TEST_RESULT=$?

# Restore original test files
echo -e "${YELLOW}Restoring original test files...${NC}"
find tests/ -type f -name "*.py" -exec sed -i '' 's/llm_provider="huggingface"/llm_provider="ollama"/g' {} \;
find tests/ -type f -name "*.py" -exec sed -i '' 's/provider="huggingface"/provider="ollama"/g' {} \;
find tests/ -type f -name "*.py" -exec sed -i '' 's/llm_model="mistralai\/Mistral-7B-Instruct-v0.2"/llm_model="llama3"/g' {} \;
find tests/ -type f -name "*.py" -exec sed -i '' 's/model_name="mistralai\/Mistral-7B-Instruct-v0.2"/model_name="llama3:latest"/g' {} \;

# Check if tests were successful
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Some tests failed. Check the test output for details.${NC}"
fi

echo -e "${GREEN}Test run complete.${NC}"
exit $TEST_RESULT 