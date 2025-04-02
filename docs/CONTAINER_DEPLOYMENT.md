# Finance RAG: Containerization Options for Free Tier Deployment

## Overview

This document outlines containerization strategies for the Finance RAG system that are compatible with free tier cloud services. The focus is on lightweight containers and minimal resource utilization.

## Containerization Approach

### Docker Configuration

#### Minimal Base Images
- **Python Application**:
  - Use `python:3.10-slim` instead of full images
  - Multi-stage builds to reduce final image size
  - Strip debug symbols and development dependencies
- **Frontend Application**:
  - Use `node:16-alpine` for build stages
  - `nginx:alpine` for serving static content
  - Distroless options for minimum attack surface

#### Optimized Dockerfile

```dockerfile
# Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /app/wheels /app/wheels
COPY --from=builder /app/requirements.txt .

# Install dependencies from wheels
RUN pip install --no-cache-dir --no-index --find-links=/app/wheels -r requirements.txt \
    && rm -rf /app/wheels

# Copy application code
COPY src/ ./src/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "src/api/app.py"]
```

### Resource Optimization

#### Memory Constraints
- **Container Limits**:
  - Set memory limits to 256MB-512MB
  - Configure Python garbage collection
  - Use memory-efficient libraries
- **CPU Allocation**:
  - Limit to 0.5-1 vCPU
  - Configure thread pools appropriately
  - Implement graceful degradation under load

#### Startup Time Optimization
- **Cold Start Reduction**:
  - Lazy loading of non-critical modules
  - Prioritize initialization steps
  - Background loading of resources
- **Warm Containers**:
  - Implement keep-alive mechanisms
  - Serverless pre-warming techniques
  - Stateless design for scaling

## Free Tier Container Hosting Options

### AWS Free Tier Options

#### AWS ECR + Lambda
- **Configuration**:
  - Store container images in ECR (free tier: 500MB storage)
  - Deploy as Lambda container images (up to 10GB)
  - Free tier: 1M requests/month and 400,000 GB-seconds compute
- **Limitations**:
  - 15-minute maximum execution time
  - Memory limitations (up to 10GB, but free tier is limited)
  - Cold start latency
- **Best Practices**:
  - Optimize container size (< 500MB)
  - Implement efficient warm-up
  - Stateless design

#### AWS App Runner (Limited Free Trial)
- **Configuration**:
  - Deploy container directly from ECR
  - Auto-scaling configuration
  - Health checks and monitoring
- **Limitations**:
  - Free trial period only
  - Limited to smaller instances
- **Best Practices**:
  - Use during development/testing
  - Plan migration path post-trial

### GCP Free Tier Options

#### Cloud Run
- **Configuration**:
  - Deploy containers directly
  - Free tier: 2M requests/month, 360,000 GB-seconds, 180,000 vCPU-seconds
  - Automatic scaling to zero
- **Limitations**:
  - Request timeout (up to 60 minutes, but consider free tier limits)
  - Memory constraints in free tier
- **Best Practices**:
  - Optimize for fast startup
  - Implement request caching
  - Use minimal container size

#### Artifact Registry
- **Configuration**:
  - Store container images (free tier: 0.5GB storage)
  - Private container registry
  - Vulnerability scanning
- **Best Practices**:
  - Regularly clean old images
  - Tag images appropriately
  - Implement image signing

### Alternative Free Options

#### Railway App
- **Configuration**:
  - Deploy containers with GitHub integration
  - Free tier: 5$ credit per month (~500 hours of compute)
  - Automatic deploys
- **Limitations**:
  - Resource constraints
  - Limited build minutes
- **Best Practices**:
  - Optimize CI/CD pipeline
  - Regular container cleanup

#### Render
- **Configuration**:
  - Deploy Docker containers
  - Free tier for static sites and web services
  - GitHub integration
- **Limitations**:
  - Free instances spin down after inactivity
  - Limited to smaller instance sizes
- **Best Practices**:
  - Implement health checks
  - Use for non-critical services

## Local Development Environment

### Docker Compose Setup

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./src:/app/src
    environment:
      - DATABASE_URL=sqlite:///./finance.db
      - VECTOR_DB_PATH=./vector_db
      - OLLAMA_API_URL=http://ollama:11434
    depends_on:
      - ollama
    # Memory and CPU limits for local testing
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 256M

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 4G

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    volumes:
      - ./frontend/src:/app/src
    depends_on:
      - api
    deploy:
      resources:
        limits:
          cpus: '0.2'
          memory: 128M

volumes:
  ollama_data:
```

### Development Workflow
- **Local Testing**:
  - Run with reduced resources to match free tier
  - Profile memory and CPU usage
  - Identify and fix resource leaks
- **Image Optimization**:
  - Regular pruning of images
  - Layer caching strategy
  - Review image sizes

## Container Security for Free Tier

### Minimizing Attack Surface
- **Reduce Image Size**:
  - Remove unnecessary tools and libraries
  - Use distroless or alpine base images
  - Implement multi-stage builds
- **Principle of Least Privilege**:
  - Run as non-root user
  - Read-only file systems where possible
  - Minimal capability sets

### Security Scanning
- **Free Scanning Tools**:
  - Trivy for vulnerability scanning
  - Docker Scout (limited free tier)
  - GitHub security scanning
- **Integration Points**:
  - Pre-commit hooks
  - CI/CD pipeline integration
  - Scheduled scans

## Deployment Strategy

### CI/CD Integration
- **GitHub Actions** (free tier):
  - Automated builds on commit
  - Container vulnerability scanning
  - Automated testing before deployment
- **Environment Separation**:
  - Development containers
  - Production containers with stricter constraints
  - Testing environment for load testing

### Free Tier Monitoring
- **Container Health Metrics**:
  - Basic CloudWatch/Cloud Monitoring (free tier)
  - Container-level health checks
  - Custom logging for resource usage
- **Performance Tracking**:
  - Request latency monitoring
  - Memory usage tracking
  - Cold start frequency

## Conclusion

Containerization offers flexibility for deployment even within free tier constraints. By focusing on minimal, optimized containers and selecting appropriate free tier hosting options, the Finance RAG system can be deployed effectively without incurring costs. Regular monitoring and optimization remain essential to stay within free tier limits. 