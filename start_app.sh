#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Finance LLM Application...${NC}"

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

# Check if llama3 model is available, pull if not
if ! curl -s http://localhost:11434/api/tags | grep -q "llama3"; then
    echo -e "${YELLOW}Pulling llama3 model...${NC}"
    ollama pull llama3
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to pull llama3 model. Please pull it manually with 'ollama pull llama3'${NC}"
        exit 1
    else
        echo -e "${GREEN}Successfully pulled llama3 model${NC}"
    fi
else
    echo -e "${GREEN}llama3 model is already available${NC}"
fi

# Run tests with Ollama as the LLM provider
echo -e "${GREEN}Running tests with Ollama...${NC}"
export OLLAMA_API_BASE="http://localhost:11434"
mkdir -p test_output

# Run the tests
python -m tests.run_tests

# Check if tests were successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Some tests failed. Check the test output for details.${NC}"
fi

# If we started Ollama, keep it running
if [ -n "$OLLAMA_PID" ]; then
    echo -e "${YELLOW}Keeping Ollama running. If you want to stop it, run 'kill $OLLAMA_PID'${NC}"
fi

echo -e "${GREEN}Done!${NC}" 