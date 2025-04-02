#!/bin/bash
set -e

# Configuration
AWS_REGION="us-east-1"
ECR_REPOSITORY_NAME="finance-rag-api"
ECS_CLUSTER_NAME="finance-rag-cluster"
ECS_SERVICE_NAME="finance-rag-service"
ECS_TASK_FAMILY="finance-rag-task"
CONTAINER_NAME="finance-rag-container"
CONTAINER_PORT=8000
DESIRED_COUNT=1

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
  echo "Error: AWS CLI is not configured. Please run 'aws configure'"
  exit 1
fi

# Load environment variables
if [ -f .env ]; then
  source .env
  echo "Loaded environment variables from .env"
else
  echo "Warning: .env file not found. Make sure environment variables are set"
fi

# Step 1: Create ECR repository if it doesn't exist
echo "Checking if ECR repository exists..."
if ! aws ecr describe-repositories --repository-names $ECR_REPOSITORY_NAME --region $AWS_REGION &> /dev/null; then
  echo "Creating ECR repository $ECR_REPOSITORY_NAME..."
  aws ecr create-repository --repository-name $ECR_REPOSITORY_NAME --region $AWS_REGION
fi

# Step 2: Get ECR repository URI
ECR_REPOSITORY_URI=$(aws ecr describe-repositories --repository-names $ECR_REPOSITORY_NAME --region $AWS_REGION --query 'repositories[0].repositoryUri' --output text)
echo "ECR Repository URI: $ECR_REPOSITORY_URI"

# Step 3: Authenticate Docker to ECR
echo "Authenticating Docker to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPOSITORY_URI

# Step 4: Build and push Docker image
echo "Building Docker image..."
docker build -t $ECR_REPOSITORY_NAME .

echo "Tagging Docker image..."
docker tag $ECR_REPOSITORY_NAME:latest $ECR_REPOSITORY_URI:latest

echo "Pushing Docker image to ECR..."
docker push $ECR_REPOSITORY_URI:latest

# Step 5: Create ECS cluster if it doesn't exist
echo "Checking if ECS cluster exists..."
if ! aws ecs describe-clusters --clusters $ECS_CLUSTER_NAME --region $AWS_REGION --query 'clusters[0].status' --output text 2>/dev/null | grep -q "ACTIVE"; then
  echo "Creating ECS cluster $ECS_CLUSTER_NAME..."
  aws ecs create-cluster --cluster-name $ECS_CLUSTER_NAME --region $AWS_REGION
else
  echo "Using existing ECS cluster $ECS_CLUSTER_NAME"
fi

# Step 6: Create IAM task execution role
TASK_EXECUTION_ROLE_NAME="ecsTaskExecutionRole"
echo "Checking if IAM task execution role exists..."
if ! aws iam get-role --role-name $TASK_EXECUTION_ROLE_NAME &> /dev/null; then
  echo "Creating IAM task execution role $TASK_EXECUTION_ROLE_NAME..."
  aws iam create-role --role-name $TASK_EXECUTION_ROLE_NAME --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {
          "Service": "ecs-tasks.amazonaws.com"
        },
        "Action": "sts:AssumeRole"
      }
    ]
  }'
  
  # Attach the AmazonECSTaskExecutionRolePolicy
  aws iam attach-role-policy --role-name $TASK_EXECUTION_ROLE_NAME --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
  
  # Add policy for Systems Manager Parameter Store access
  aws iam attach-role-policy --role-name $TASK_EXECUTION_ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonSSMReadOnlyAccess
fi

# Get the role ARN
TASK_EXECUTION_ROLE_ARN=$(aws iam get-role --role-name $TASK_EXECUTION_ROLE_NAME --query 'Role.Arn' --output text)

# Store sensitive parameters in SSM Parameter Store
echo "Storing Hugging Face API key in SSM Parameter Store..."
aws ssm put-parameter \
  --name "/finance-rag/HUGGINGFACE_API_KEY" \
  --type "SecureString" \
  --value "${HUGGINGFACE_API_KEY}" \
  --overwrite \
  --region $AWS_REGION

# If MONGODB_URI is set, also store it in SSM
if [ ! -z "$MONGODB_URI" ]; then
  echo "Storing MongoDB URI in SSM Parameter Store..."
  aws ssm put-parameter \
    --name "/finance-rag/MONGODB_URI" \
    --type "SecureString" \
    --value "${MONGODB_URI}" \
    --overwrite \
    --region $AWS_REGION
fi

# Step 7: Register ECS task definition
echo "Registering ECS task definition..."
# Create task definition JSON
cat > task-definition.json << EOL
{
  "family": "${ECS_TASK_FAMILY}",
  "networkMode": "awsvpc",
  "executionRoleArn": "${TASK_EXECUTION_ROLE_ARN}",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [
    {
      "name": "${CONTAINER_NAME}",
      "image": "${ECR_REPOSITORY_URI}:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": ${CONTAINER_PORT},
          "hostPort": ${CONTAINER_PORT},
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "PYTHONPATH",
          "value": "/app"
        },
        {
          "name": "DEFAULT_LLM_PROVIDER",
          "value": "huggingface"
        },
        {
          "name": "DEFAULT_LLM_MODEL",
          "value": "mistralai/Mistral-7B-Instruct-v0.2"
        },
        {
          "name": "CLOUD_DEPLOYMENT",
          "value": "true"
        }
      ],
      "secrets": [
        {
          "name": "HUGGINGFACE_API_KEY",
          "valueFrom": "arn:aws:ssm:${AWS_REGION}:$(aws sts get-caller-identity --query 'Account' --output text):parameter/finance-rag/HUGGINGFACE_API_KEY"
        }
EOL

# Add MongoDB URI if it's set
if [ ! -z "$MONGODB_URI" ]; then
  cat >> task-definition.json << EOL
        ,
        {
          "name": "MONGODB_URI",
          "valueFrom": "arn:aws:ssm:${AWS_REGION}:$(aws sts get-caller-identity --query 'Account' --output text):parameter/finance-rag/MONGODB_URI"
        }
EOL
fi

# Close the secrets array and add the rest of the definition
cat >> task-definition.json << EOL
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/${ECS_TASK_FAMILY}",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
EOL

# Register the task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json --region $AWS_REGION

# Step 8: Create security group for the service
echo "Creating security group for the service..."
SG_NAME="finance-rag-sg"

# Check if security group already exists
SG_ID=$(aws ec2 describe-security-groups --filters Name=group-name,Values=$SG_NAME --query 'SecurityGroups[0].GroupId' --output text --region $AWS_REGION)

if [ "$SG_ID" == "None" ]; then
  # Get default VPC ID
  VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text --region $AWS_REGION)
  
  # Create security group
  SG_ID=$(aws ec2 create-security-group --group-name $SG_NAME --description "Security group for Finance RAG API" --vpc-id $VPC_ID --region $AWS_REGION --query 'GroupId' --output text)
  
  # Add inbound rules for HTTP and HTTPS
  aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 80 --cidr 0.0.0.0/0 --region $AWS_REGION
  aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 443 --cidr 0.0.0.0/0 --region $AWS_REGION
  aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 8000 --cidr 0.0.0.0/0 --region $AWS_REGION
  
  echo "Created security group $SG_ID"
else
  echo "Using existing security group $SG_ID"
fi

# Step 9: Get public subnets from default VPC
echo "Getting public subnets from default VPC..."
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text --region $AWS_REGION)

# Get all subnets in the default VPC
SUBNET_IDS=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID --query 'Subnets[*].SubnetId' --output text --region $AWS_REGION)

# If we couldn't get any subnets, provide an error message
if [ -z "$SUBNET_IDS" ]; then
    echo "Error: No subnets found in the default VPC. Please specify subnet IDs manually."
    read -p "Enter at least one subnet ID (or press Enter to exit): " manual_subnet
    if [ -z "$manual_subnet" ]; then
        exit 1
    fi
    SUBNET_IDS=$manual_subnet
fi

# Convert space-separated list to comma-separated
SUBNET_IDS_CSV=$(echo $SUBNET_IDS | tr ' ' ',')
echo "Using subnets: $SUBNET_IDS_CSV"

# Step 10: Create ECS service
echo "Creating ECS service..."
# Check if service already exists
if aws ecs describe-services --cluster $ECS_CLUSTER_NAME --services $ECS_SERVICE_NAME --region $AWS_REGION --query 'services[0].status' --output text 2>/dev/null | grep -q "ACTIVE"; then
  echo "Updating existing ECS service $ECS_SERVICE_NAME..."
  aws ecs update-service \
      --cluster $ECS_CLUSTER_NAME \
      --service $ECS_SERVICE_NAME \
      --task-definition $ECS_TASK_FAMILY \
      --desired-count $DESIRED_COUNT \
      --region $AWS_REGION
else
  echo "Creating new ECS service $ECS_SERVICE_NAME..."
  aws ecs create-service \
    --cluster $ECS_CLUSTER_NAME \
    --service-name $ECS_SERVICE_NAME \
    --task-definition $ECS_TASK_FAMILY \
    --desired-count $DESIRED_COUNT \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS_CSV],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
    --region $AWS_REGION
fi

# Step 11: Create CloudWatch Logs group
echo "Creating CloudWatch Logs group..."
aws logs create-log-group --log-group-name /ecs/$ECS_TASK_FAMILY --region $AWS_REGION || true

# Step 12: Get the public IP of the task
echo "Waiting for service to stabilize and get public IP..."
aws ecs wait services-stable --cluster $ECS_CLUSTER_NAME --services $ECS_SERVICE_NAME --region $AWS_REGION

TASK_ARN=$(aws ecs list-tasks --cluster $ECS_CLUSTER_NAME --service-name $ECS_SERVICE_NAME --region $AWS_REGION --query 'taskArns[0]' --output text)
TASK_DETAILS=$(aws ecs describe-tasks --cluster $ECS_CLUSTER_NAME --tasks $TASK_ARN --region $AWS_REGION)
ENI_ID=$(echo $TASK_DETAILS | jq -r '.tasks[0].attachments[0].details[] | select(.name == "networkInterfaceId") | .value')
PUBLIC_IP=$(aws ec2 describe-network-interfaces --network-interface-ids $ENI_ID --region $AWS_REGION --query 'NetworkInterfaces[0].Association.PublicIp' --output text)

echo "Deployment completed!"
echo "Your API is available at: http://$PUBLIC_IP:8000"
echo "Health endpoint: http://$PUBLIC_IP:8000/health"

# Clean up temporary files
rm -f task-definition.json 