# Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Build wheels
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Final stage
FROM python:3.10-slim

# Create non-root user
RUN groupadd -r financerag && useradd -r -g financerag financerag

WORKDIR /app

# Copy wheels from builder stage
COPY --from=builder /app/wheels /app/wheels
COPY --from=builder /app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --no-index --find-links=/app/wheels -r requirements.txt \
    && rm -rf /app/wheels

# Copy application code
COPY src/ ./src/

# Add startup script
COPY src/infrastructure/entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create directories and set permissions
RUN mkdir -p /app/data /app/logs \
    && chown -R financerag:financerag /app

# Switch to non-root user
USER financerag

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=sqlite:///./data/finance.db
ENV VECTOR_DB_PATH=./data/vector_db
ENV LOG_LEVEL=INFO

# Expose port
EXPOSE 8000

# Set health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
ENTRYPOINT ["/app/entrypoint.sh"] 