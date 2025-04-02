# Finance RAG System Deployment Guide

This guide provides step-by-step instructions for deploying the Finance RAG system to AWS using free tier services.

## Prerequisites

Before you begin, make sure you have the following:

1. **AWS Account**
   - Free tier eligible account
   - IAM user with programmatic access
   - AWS CLI installed and configured

2. **MongoDB Atlas Account**
   - Free tier M0 cluster created
   - Database user with read/write privileges
   - Network access configured (IP whitelist)

3. **Node.js and npm**
   - Node.js 14.x or higher
   - npm 6.x or higher

4. **Python Environment**
   - Python 3.8+
   - pip package manager

## Step 1: Configure Environment

1. **Clone the repository and navigate to the project directory**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   npm install
   ```

3. **Set up environment variables**
   - Copy `.env.sample` to `.env`
   - Edit `.env` and add your MongoDB Atlas connection string
   ```
   MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/finance_rag?retryWrites=true&w=majority
   ```

4. **Configure AWS CLI**
   ```bash
   aws configure
   ```
   Enter your AWS Access Key ID, Secret Access Key, default region (e.g., us-east-1), and output format (json).

## Step 2: MongoDB Atlas Setup

1. **Initialize MongoDB Collections**
   ```bash
   python src/db/mongodb_init.py
   ```
   This script creates the necessary collections and indexes in your MongoDB Atlas database.

2. **Import sample data (optional)**
   ```bash
   python ingest_data.py --mongodb
   ```

## Step 3: Deploy Backend API

1. **Using the Deployment Script**
   
   For a one-step deployment:
   ```bash
   ./deploy.sh
   ```
   
   Or specify stage and region:
   ```bash
   ./deploy.sh --stage prod --region us-west-2
   ```

2. **Manual Deployment**
   
   If you prefer a manual deployment:
   ```bash
   cd src/infrastructure
   serverless deploy --stage dev --verbose
   ```

## Step 4: Frontend Static Build and Deployment

### Building the Frontend

1. **Optimize and Build the Frontend**
   ```bash
   cd frontend
   npm install
   npm run build
   ```
   
   This creates an optimized production build in the `frontend/dist` directory with:
   - Minified JavaScript bundles
   - Optimized CSS
   - Compressed assets
   - Tree-shaking for minimal bundle size

2. **Environment Configuration**
   
   Create a `.env.production` file with your API endpoint:
   ```
   VITE_API_ENDPOINT=https://your-api-id.execute-api.region.amazonaws.com/dev
   ```

### Deploying to Vercel (Free Tier)

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy to Vercel**
   ```bash
   cd frontend
   vercel --prod
   ```

4. **Configure Environment Variables in Vercel**
   - Go to your Vercel project dashboard
   - Navigate to Settings > Environment Variables
   - Add `VITE_API_ENDPOINT` with your API URL

### Deploying to Netlify (Free Tier)

1. **Install Netlify CLI**
   ```bash
   npm install -g netlify-cli
   ```

2. **Login to Netlify**
   ```bash
   netlify login
   ```

3. **Create netlify.toml Configuration**
   
   Create a `netlify.toml` file in the frontend directory:
   ```toml
   [build]
     publish = "dist"
     command = "npm run build"
   
   [[redirects]]
     from = "/*"
     to = "/index.html"
     status = 200
   
   [build.environment]
     NODE_VERSION = "16"
   ```

4. **Deploy to Netlify**
   ```bash
   cd frontend
   netlify deploy --prod
   ```

5. **Configure Environment Variables in Netlify**
   - Go to your Netlify site dashboard
   - Navigate to Site settings > Build & deploy > Environment
   - Add `VITE_API_ENDPOINT` with your API URL

### CDN Configuration

Both Vercel and Netlify automatically provide CDN services for your static assets. For optimal performance:

1. **Cache Control Headers**
   
   For Netlify, add the following to `netlify.toml`:
   ```toml
   [[headers]]
     for = "/assets/*"
     [headers.values]
       Cache-Control = "public, max-age=31536000, immutable"
   ```

2. **Optimized Assets**
   
   The build process automatically:
   - Applies content hashing for cache busting
   - Compresses images and static assets
   - Implements code splitting for faster load times

## Step 5: Verify Deployment

1. **Test API Endpoints**
   ```bash
   curl -X GET https://your-api-id.execute-api.region.amazonaws.com/dev/health
   ```

2. **Test Frontend Application**
   - Open your Vercel or Netlify URL in a browser
   - Verify that the application loads correctly
   - Test the chat interface and file upload functionality

3. **Monitor CloudWatch Logs**
   - Check the CloudWatch Log Groups for your Lambda functions
   - Verify that logs are being generated correctly
   - Look for any errors or warnings

## Step 6: Set Up Monitoring

1. **Configure CloudWatch Alarms**
   ```bash
   aws cloudwatch put-metric-alarm --alarm-name api-errors --alarm-description "Alarm for API errors" \
   --metric-name Errors --namespace AWS/Lambda --statistic Sum --period 60 --threshold 1 \
   --comparison-operator GreaterThanOrEqualToThreshold --evaluation-periods 1 \
   --alarm-actions arn:aws:sns:region:account-id:topic-name \
   --dimensions Name=FunctionName,Value=your-function-name
   ```

2. **Set Up Health Checks**
   ```bash
   aws route53 create-health-check --caller-reference $(date +%s) \
   --health-check-config Type=HTTPS,FullyQualifiedDomainName=your-api-id.execute-api.region.amazonaws.com,Port=443,ResourcePath=/dev/health
   ```

## Cleanup

To remove all deployed resources when they're no longer needed:

1. **Remove API and Lambda Functions**
   ```bash
   cd src/infrastructure
   serverless remove --stage dev
   ```

2. **Remove Frontend Deployment**
   - Vercel: `vercel remove finance-rag`
   - Netlify: `netlify site:delete`

3. **Remove CloudWatch Alarms**
   ```bash
   aws cloudwatch delete-alarms --alarm-names api-errors
   ```

4. **Remove Health Checks**
   ```bash
   aws route53 delete-health-check --health-check-id health-check-id
   ```

## Troubleshooting

### Common Issues

1. **AWS Lambda Deployment Failure**
   - Check IAM permissions
   - Verify AWS region configuration
   - Ensure Lambda function size is within limits

2. **MongoDB Connection Issues**
   - Verify MongoDB Atlas connection string
   - Check IP whitelist in MongoDB Atlas
   - Ensure correct database name in connection string

3. **Cold Start Performance**
   - Consider increasing Lambda memory allocation
   - Implement Lambda warm-up strategies
   - Optimize code for faster initialization

### Logs and Debugging

1. **View Lambda Logs**
   ```bash
   aws logs filter-log-events \
     --log-group-name /aws/lambda/finance-rag-api-dev-query \
     --start-time $(date -v-1d +%s)000
   ```

2. **View API Gateway Logs**
   ```bash
   aws logs filter-log-events \
     --log-group-name API-Gateway-Execution-Logs_api-id/dev
   ```

## Security Considerations

1. **API Authentication**
   - Consider implementing API key authentication
   - Add JWT-based authentication for user-specific data

2. **Database Security**
   - Use strong, unique passwords for MongoDB Atlas
   - Limit IP access in MongoDB Atlas
   - Enable MongoDB Atlas encryption at rest

3. **Environment Variables**
   - Store sensitive information in AWS Secrets Manager
   - Don't commit `.env` files to source control

## Free Tier Optimization

To stay within free tier limits:

1. **AWS Lambda**
   - Keep memory allocation at 256MB or lower
   - Implement efficient cold start strategies
   - Cache responses when possible

2. **MongoDB Atlas**
   - Implement TTL indexes for automatic data cleanup
   - Monitor storage usage (limit: 512MB)
   - Compress data where possible

3. **API Gateway**
   - Be mindful of the 1M requests/month limit
   - Implement client-side caching
   - Consider implementing request throttling

## Next Steps

1. **Implement Authentication**
   - See `docs/AUTHENTICATION.md` for details

2. **Set Up CI/CD Pipeline**
   - Configure GitHub Actions workflows for automated deployment
   - See `.github/workflows/deploy.yml` for a template

3. **Enhance Monitoring**
   - Set up CloudWatch dashboards
   - Configure alerts for critical issues 