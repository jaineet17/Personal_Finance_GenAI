#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Finance LLM App Test Runner${NC}"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}Error: Ollama is not installed. Please install Ollama first:${NC}"
    echo -e "${YELLOW}Visit https://ollama.com/ for installation instructions.${NC}"
    exit 1
fi

# Check if Ollama is running
if ! curl --silent --output /dev/null --fail http://localhost:11434/api/tags; then
    echo -e "${YELLOW}Ollama is not running. Starting Ollama...${NC}"
    
    # Start Ollama in the background
    ollama serve &
    
    # Save the PID to kill it later if needed
    OLLAMA_PID=$!
    
    # Wait for Ollama to start
    echo -e "${YELLOW}Waiting for Ollama to start...${NC}"
    for i in {1..30}; do
        if curl --silent --output /dev/null --fail http://localhost:11434/api/tags; then
            echo -e "${GREEN}Ollama started successfully!${NC}"
            break
        fi
        
        if [ $i -eq 30 ]; then
            echo -e "${RED}Timed out waiting for Ollama to start. Please start Ollama manually.${NC}"
            exit 1
        fi
        
        echo -n "."
        sleep 1
    done
else
    echo -e "${GREEN}Ollama is already running.${NC}"
fi

# Check available models
MODELS_JSON=$(curl -s http://localhost:11434/api/tags)
echo -e "${YELLOW}Available models:${NC}"
echo "$MODELS_JSON" | grep -o '"name":"[^"]*"' | cut -d'"' -f4

# Check if llama3 model (with any tag) is available, pull if not
if ! echo "$MODELS_JSON" | grep -q "llama3"; then
    echo -e "${YELLOW}No llama3 model found. Pulling llama3:latest model (this may take some time)...${NC}"
    ollama pull llama3:latest
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to pull llama3:latest model. Please pull it manually with 'ollama pull llama3:latest'${NC}"
        exit 1
    else
        echo -e "${GREEN}Successfully pulled llama3:latest model${NC}"
    fi
else
    echo -e "${GREEN}llama3 model is available${NC}"
    # Get the full model name including tag
    LLAMA3_MODEL=$(echo "$MODELS_JSON" | grep -o '"name":"llama3[^"]*"' | cut -d'"' -f4 | head -1)
    echo -e "${GREEN}Using model: ${LLAMA3_MODEL}${NC}"
fi

# Set environment variables for testing
export OLLAMA_API_BASE="http://localhost:11434"
export ENVIRONMENT="test"
# Ensure we use the correct model name with tag
export OLLAMA_MODEL="llama3:latest"

# Create output directories
mkdir -p test_output

# Run the tests
echo -e "${GREEN}Running tests with Ollama...${NC}"
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

# Check if tests were successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Some tests failed. Check the test output for details.${NC}"
fi

# Clean up
if [ -n "$OLLAMA_PID" ]; then
    echo -e "${YELLOW}Ollama was started by this script. Do you want to stop it? (y/n)${NC}"
    read -r stop_ollama
    if [[ "$stop_ollama" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Stopping Ollama...${NC}"
        kill $OLLAMA_PID
        echo -e "${GREEN}Ollama stopped.${NC}"
    else
        echo -e "${YELLOW}Keeping Ollama running. You can stop it later with 'kill $OLLAMA_PID'${NC}"
    fi
fi

echo -e "${GREEN}Test run complete.${NC}" 