#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to display messages
function echo_step() {
  echo -e "${GREEN}[DEPLOY]${NC} $1"
}

function echo_warn() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

function echo_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Check if AWS CLI is configured
echo_step "Checking AWS configuration..."
if ! aws sts get-caller-identity &> /dev/null; then
  echo_error "AWS CLI is not configured correctly. Please run 'aws configure' first."
  exit 1
fi

# Check if MongoDB URI is set
if ! grep -q "^MONGODB_URI=mongodb+srv://" .env; then
  echo_error "MongoDB URI is not set correctly in .env file."
  echo "Please update the .env file with your MongoDB Atlas connection string."
  exit 1
fi

# Parse command line arguments
STAGE="dev"
REGION="us-east-1"

while [[ $# -gt 0 ]]; do
  case $1 in
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    *)
      echo_error "Unknown option: $1"
      echo "Usage: $0 [--stage dev|prod] [--region aws-region]"
      exit 1
      ;;
  esac
done

echo_step "Deploying to AWS (Stage: $STAGE, Region: $REGION)..."

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
  echo_step "Installing npm dependencies..."
  npm install
fi

# Install serverless plugins if needed
if [ ! -d "node_modules/serverless-python-requirements" ]; then
  echo_step "Installing serverless plugins..."
  npm install --save-dev serverless-python-requirements serverless-dotenv-plugin
fi

# Create layers directory if it doesn't exist
mkdir -p src/infrastructure/layers/python

# Create database in MongoDB Atlas
echo_step "Initializing MongoDB collections and indexes..."
python src/db/mongodb_init.py

# Deploy backend with Serverless Framework
echo_step "Deploying backend to AWS Lambda..."
cd src/infrastructure
serverless deploy --stage $STAGE --region $REGION --verbose

# Get the API endpoint
API_ENDPOINT=$(serverless info --stage $STAGE --verbose | grep -o 'https://[^[:space:]]*.amazonaws.com/dev')
if [ -z "$API_ENDPOINT" ]; then
  echo_warn "Could not automatically detect API endpoint."
  echo "Please check your deployment and update the frontend configuration manually."
else
  echo_step "API deployed to: $API_ENDPOINT"
  
  # Update frontend configuration
  echo_step "Updating frontend configuration with API endpoint..."
  cd ../../frontend
  
  # Create .env file for frontend
  echo "REACT_APP_API_URL=$API_ENDPOINT" > .env
  
  # Build frontend
  echo_step "Building frontend..."
  if [ -f "package.json" ]; then
    npm install
    npm run build
  else
    echo_warn "Frontend package.json not found. Skipping frontend build."
  fi
fi

echo_step "Deployment completed!"
echo ""
echo "Next steps:"
echo "1. If frontend build was successful, deploy the frontend to your hosting provider"
echo "   (e.g., Netlify, Vercel, or AWS S3)"
echo "2. Test your deployed application"
echo "3. Set up monitoring and alerts in AWS CloudWatch"
echo "" 