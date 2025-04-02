#!/bin/bash

# Set default port
PORT=${FRONTEND_PORT:-3001}

# Set default host
HOST=${FRONTEND_HOST:-127.0.0.1}

echo "Starting Finance LLM Assistant frontend on http://$HOST:$PORT"

# Change to the frontend directory and start Python's HTTP server
cd frontend && python -m http.server $PORT --bind $HOST 