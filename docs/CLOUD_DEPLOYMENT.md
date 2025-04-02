# Finance RAG System: Free Tier Cloud Deployment Guide

## Overview

This document outlines the strategy for deploying the Finance RAG system to cloud infrastructure using exclusively free tier services. The deployment focuses on minimizing costs while maintaining system functionality and security.

## Free Tier Cloud Provider Analysis

### AWS Free Tier Options
- **Lambda**: 1M free requests per month and 400,000 GB-seconds of compute time
- **API Gateway**: 1M API calls per month
- **DynamoDB**: 25GB of storage with limited read/write capacity
- **S3**: 5GB of standard storage, 20,000 GET requests, 2,000 PUT requests
- **EC2**: 750 hours per month of t2.micro instance usage (for 12 months)

### GCP Free Tier Options
- **Cloud Functions**: 2M invocations per month, 400,000 GB-seconds, 200,000 GHz-seconds
- **Cloud Run**: 2M requests per month, 360,000 GB-seconds, 180,000 vCPU-seconds
- **App Engine**: 28 instance hours per day
- **Firestore**: 1GB storage, 50,000 reads, 20,000 writes, 20,000 deletes per day
- **Cloud Storage**: 5GB of regional storage, 5,000 Class A operations per month

### Other Providers
- **Vercel/Netlify**: Free tier for static frontend hosting
- **MongoDB Atlas**: 512MB free cluster
- **Railway**: Limited free tier for containerized applications
- **Heroku**: Limited free dyno hours (may sleep after 30 minutes of inactivity)

## Selected Architecture

Based on free tier limitations, we'll implement a **serverless hybrid architecture**:

1. **Frontend**: Static site hosted on Vercel/Netlify
2. **API Layer**: Serverless functions on AWS Lambda or GCP Cloud Functions
3. **Database**: MongoDB Atlas free tier for vector database
4. **LLM Integration**: Self-hosted Ollama or API connection to free LLM providers
5. **File Storage**: Minimal object storage with AWS S3 or GCP Cloud Storage

## Infrastructure as Code

We'll use infrastructure as code (IaC) to define our cloud resources using either:
- AWS CDK/CloudFormation for AWS resources
- Terraform for multi-cloud deployment
- Serverless Framework for simplified serverless deployment

## Serverless Deployment Plan

### API Deployment
1. **Package API code**: Create lightweight packages for serverless functions
2. **Configure API routes**: Map HTTP endpoints to serverless functions
3. **Implement authentication**: Use JWT tokens with minimal overhead
4. **Set up database connections**: Optimize database connections for serverless environment

### Frontend Deployment
1. **Build static assets**: Optimize frontend for minimal size
2. **Configure routes**: Implement client-side routing
3. **Set up CDN**: Utilize free CDN services from hosting provider

### Database Setup
1. **Configure MongoDB Atlas**: Set up free tier cluster
2. **Implement data model**: Optimize schema for limited storage
3. **Create indexes**: Configure indexes for query performance
4. **Set up backup strategy**: Implement regular exports within free limits

## Free Tier Resource Optimization

### Memory Usage Optimization
- Minimize Lambda/Function memory allocation
- Implement efficient data loading patterns
- Use compression for stored data

### Storage Optimization
- Implement TTL for temporary data
- Use tiered storage approach
- Clean up unused resources automatically

### Compute Optimization
- Optimize cold start times
- Implement efficient caching
- Use appropriate instance sizes

## Monitoring & Operations

### Free Monitoring Tools
- CloudWatch (AWS) limited metrics
- Cloud Monitoring (GCP) limited metrics
- Uptime Robot for basic uptime monitoring
- Custom logging solutions

### Deployment Workflow
- GitHub Actions for CI/CD (free tier)
- Manual approval for production deployments
- Rollback procedures

## Security Considerations

- JWT-based authentication
- Environment variable management
- Minimal IAM permissions
- API rate limiting
- Input validation

## Next Steps

1. Set up infrastructure as code repositories
2. Create deployment scripts
3. Configure CI/CD pipelines
4. Implement monitoring and alerting
5. Document operation procedures

## Cloud Deployment Architecture for Finance RAG

This document outlines the cloud architecture for deploying the Finance RAG application using AWS free tier services.

## Architecture Overview

![Finance RAG Cloud Architecture](../assets/images/cloud_architecture.png)

The Finance RAG application uses a serverless architecture to stay within free tier limits:

1. **Frontend**: Static website hosted on Vercel or Netlify (free tier)
2. **API Backend**: AWS Lambda with API Gateway or ECS Fargate
3. **Database**: MongoDB Atlas M0 free cluster (512MB)
4. **Vector Storage**: Optimized storage in MongoDB Atlas
5. **File Storage**: AWS S3 (within 5GB free tier limit)

## Free Tier Service Selection

| Component | Service | Free Tier Limit | Our Usage |
|-----------|---------|-----------------|-----------|
| Frontend | Vercel/Netlify | Unlimited static sites | 1 site |
| Backend API | AWS Lambda | 1M requests/month | ~10K req/month |
| API Gateway | AWS API Gateway | 1M requests/month | ~10K req/month |
| Database | MongoDB Atlas | 512MB storage | ~350MB |
| File Storage | AWS S3 | 5GB storage | ~1GB |
| Monitoring | CloudWatch | 5GB logs/month | ~2GB logs |
| Container Registry | ECR | 500MB storage | ~300MB |

## Deployment Strategies

The Finance RAG application can be deployed using:

1. **Serverless Deployment** (AWS Lambda + API Gateway)
   - Lower cost for sporadic usage
   - Automatic scaling to zero
   - Cold start considerations

2. **Container Deployment** (AWS ECS Fargate)
   - More consistent performance
   - Better for steady usage patterns
   - Requires container registry

Both approaches are documented in this repository.

## CI/CD Pipeline with GitHub Actions

We've implemented a comprehensive CI/CD pipeline using GitHub Actions to automate testing, building, and deployment of the Finance RAG application.

### Pipeline Configuration

The CI/CD workflow is defined in the `.github/workflows` directory with the following files:

1. **ci.yml**: Continuous Integration workflow
2. **cd-dev.yml**: Continuous Deployment to development environment
3. **cd-prod.yml**: Continuous Deployment to production environment

### Continuous Integration Workflow

The CI workflow runs on every push to any branch and on pull requests to main/master:

```yaml
name: CI

on:
  push:
    branches: [ '**' ]
  pull_request:
    branches: [ main, master ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Lint with flake8
        run: flake8 src tests
      
      - name: Run unit tests
        run: pytest tests/unit
      
      - name: Run integration tests
        run: pytest tests/integration
        env:
          DEFAULT_LLM_PROVIDER: huggingface
          HUGGINGFACE_API_KEY: ${{ secrets.HUGGINGFACE_API_KEY }}
```

### Development Deployment Workflow

The development deployment workflow runs on pushes to the dev branch:

```yaml
name: Deploy to Development

on:
  push:
    branches: [ dev ]

jobs:
  deploy-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          npm install -g serverless
          npm install
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Deploy to AWS Lambda
        run: |
          cd src/infrastructure
          serverless deploy --stage dev
        env:
          MONGODB_URI: ${{ secrets.MONGODB_URI_DEV }}
          HUGGINGFACE_API_KEY: ${{ secrets.HUGGINGFACE_API_KEY }}
      
      - name: Get deployed API URL
        id: get-api-url
        run: |
          cd src/infrastructure
          API_URL=$(serverless info --stage dev --verbose | grep -o 'https://[^[:space:]]*.amazonaws.com/dev')
          echo "::set-output name=api_url::$API_URL"
  
  deploy-frontend:
    needs: deploy-backend
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '16'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json
      
      - name: Install dependencies
        run: |
          cd frontend
          npm install
      
      - name: Build frontend
        run: |
          cd frontend
          echo "VITE_API_ENDPOINT=${{ needs.deploy-backend.outputs.api_url }}" > .env.production
          npm run build
      
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          working-directory: ./frontend
```

### Production Deployment Workflow

The production deployment workflow runs manually or on releases:

```yaml
name: Deploy to Production

on:
  release:
    types: [published]
  workflow_dispatch:

jobs:
  deploy-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          npm install -g serverless
          npm install
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Deploy to AWS Lambda
        run: |
          cd src/infrastructure
          serverless deploy --stage prod
        env:
          MONGODB_URI: ${{ secrets.MONGODB_URI_PROD }}
          HUGGINGFACE_API_KEY: ${{ secrets.HUGGINGFACE_API_KEY }}
      
      - name: Get deployed API URL
        id: get-api-url
        run: |
          cd src/infrastructure
          API_URL=$(serverless info --stage prod --verbose | grep -o 'https://[^[:space:]]*.amazonaws.com/prod')
          echo "::set-output name=api_url::$API_URL"
  
  deploy-frontend:
    needs: deploy-backend
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '16'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json
      
      - name: Install dependencies
        run: |
          cd frontend
          npm install
      
      - name: Build frontend
        run: |
          cd frontend
          echo "VITE_API_ENDPOINT=${{ needs.deploy-backend.outputs.api_url }}" > .env.production
          npm run build
      
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          working-directory: ./frontend
          vercel-args: '--prod'
```

### Required Secrets

The following secrets need to be configured in the GitHub repository settings:

| Secret | Description |
|--------|-------------|
| AWS_ACCESS_KEY_ID | AWS access key with deployment permissions |
| AWS_SECRET_ACCESS_KEY | AWS secret key |
| HUGGINGFACE_API_KEY | API key for Hugging Face |
| MONGODB_URI_DEV | MongoDB Atlas URI for development |
| MONGODB_URI_PROD | MongoDB Atlas URI for production |
| VERCEL_TOKEN | Vercel API token |
| VERCEL_PROJECT_ID | Vercel project ID |
| VERCEL_ORG_ID | Vercel organization ID |

## Infrastructure as Code

For more advanced deployment scenarios, we also provide Terraform configurations in the `terraform` directory. These configurations define all AWS resources used by the Finance RAG application.

The Terraform setup includes:
- API Gateway and Lambda functions
- IAM roles and policies
- CloudWatch log groups
- S3 buckets
- Security groups

See the `terraform/README.md` file for details on how to use these configurations. 