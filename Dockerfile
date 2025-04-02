FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables
ENV PYTHONPATH=/app
ENV DEFAULT_LLM_PROVIDER=huggingface
ENV DEFAULT_LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
# This is intentionally set but will be ignored as we've hardcoded
# to use the real implementation in the code
ENV CLOUD_DEPLOYMENT=false

# Expose the API port
EXPOSE 8000

# Use bash to execute uvicorn (to help with exec format error)
CMD ["/bin/bash", "-c", "python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000"] 