#!/bin/bash

# Finance RAG Application Runner

# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Check if .env file exists, if not copy from sample
if [ ! -f ".env" ]; then
    if [ -f ".env.sample" ]; then
        echo "Creating .env file from sample..."
        cp .env.sample .env
        echo "Created .env file. Please edit it with your API keys."
    else
        echo "Warning: No .env or .env.sample file found."
    fi
fi

# Execute the CLI with any provided arguments
python src/cli/rag_cli.py "$@" 