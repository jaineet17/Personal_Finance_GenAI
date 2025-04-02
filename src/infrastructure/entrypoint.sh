#!/bin/bash
set -e

# Print environment (hide sensitive data)
echo "Starting Finance RAG API with configuration:"
echo "- Database Path: $DATABASE_URL"
echo "- Vector DB Path: $VECTOR_DB_PATH"
echo "- Log Level: $LOG_LEVEL"

# Ensure directories exist
mkdir -p $(dirname "$DATABASE_URL" | sed 's/sqlite:\/\///')
mkdir -p "$VECTOR_DB_PATH"

# Function to handle graceful shutdown
function handle_sigterm() {
  echo "Received SIGTERM, shutting down gracefully"
  # Add any cleanup tasks here
  exit 0
}

# Register signal handler
trap handle_sigterm SIGTERM

# Initialize database if needed
echo "Checking database..."
if [ ! -f "${DATABASE_URL#sqlite:///}" ]; then
  echo "Initializing database..."
  python -m src.db.initialize_db
fi

# Check for Ollama connection
if [ ! -z "$OLLAMA_API_URL" ]; then
  echo "Ollama API URL configured: $OLLAMA_API_URL"
  # Optional: add health check for Ollama connection
fi

# Start application with Gunicorn (optimized for free tier)
echo "Starting Finance RAG API server..."
exec gunicorn \
  --bind 0.0.0.0:8000 \
  --workers 1 \
  --threads 2 \
  --timeout 120 \
  --worker-class uvicorn.workers.UvicornWorker \
  --log-level $LOG_LEVEL \
  --access-logfile - \
  --error-logfile - \
  --forwarded-allow-ips '*' \
  src.api.app:app 