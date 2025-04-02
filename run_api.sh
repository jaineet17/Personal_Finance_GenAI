#!/bin/bash

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    export $(grep -v '^#' .env | xargs)
fi

# Make sure we're using the correct Python
if [ -d "venv" ]; then
    echo "Activating virtual environment"
    source venv/bin/activate
fi

# Set default port
PORT=${API_PORT:-8000}

# Set default host
HOST=${API_HOST:-127.0.0.1}

# Set default environment
ENVIRONMENT=${ENVIRONMENT:-development}

echo "Starting Finance LLM API server on $HOST:$PORT in $ENVIRONMENT mode"

# Run the API server with hot reload in development
if [ "$ENVIRONMENT" = "development" ]; then
    uvicorn src.api.app:app --host $HOST --port $PORT --reload
else
    # For production, don't use reload
    uvicorn src.api.app:app --host $HOST --port $PORT
fi 