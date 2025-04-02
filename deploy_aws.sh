#!/bin/bash
set -e

echo "Deploying Finance RAG API to AWS Lambda (dev stage)..."

# MongoDB connection string (URL encoded)
MONGODB_URI="mongodb+srv://username:password@cluster.example.mongodb.net/finance_rag?retryWrites=true&w=majority"

# Hugging Face API key
HUGGINGFACE_API_KEY="your_huggingface_api_key_here"

# Check if requirements-lambda.txt exists, create if not
if [ ! -f "requirements-lambda.txt" ]; then
  echo "Creating minimal requirements-lambda.txt..."
  cat > requirements-lambda.txt << EOL
# Minimal requirements for Lambda deployment
fastapi>=0.100.0
mangum>=0.17.0
pydantic>=2.0.0
pymongo>=4.0.0
python-dotenv>=1.0.0
requests>=2.31.0
huggingface-hub>=0.17.3
EOL
fi

# Make sure serverless plugins are installed
if [ ! -d "node_modules/serverless-python-requirements" ]; then
  echo "Installing serverless plugins..."
  npm install --save-dev serverless-python-requirements
fi

# Clean up any previous deployment artifacts
echo "Cleaning previous deployment artifacts..."
rm -rf .serverless || true
rm -rf node_modules/.cache || true

# Deploy with environment variables set on command line
echo "Starting deployment..."
export MONGODB_URI="$MONGODB_URI"
export DATABASE_URL="$MONGODB_URI"
export VECTOR_DB_PATH="./vector_db"
export STAGE="dev"
export HUGGINGFACE_API_KEY="$HUGGINGFACE_API_KEY"

# Run serverless deploy
echo "Deploying with optimized configuration..."
serverless deploy --verbose

echo "Deployment completed!" 