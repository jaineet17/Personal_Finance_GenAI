terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }

  required_version = ">= 1.2.0"
}

provider "aws" {
  region = var.aws_region
}

# S3 bucket for frontend static hosting (free tier)
resource "aws_s3_bucket" "frontend" {
  bucket = var.frontend_bucket_name

  tags = {
    Name        = "Finance RAG Frontend"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_website_configuration" "frontend" {
  bucket = aws_s3_bucket.frontend.bucket

  index_document {
    suffix = "index.html"
  }

  error_document {
    key = "error.html"
  }
}

resource "aws_s3_bucket_public_access_block" "frontend" {
  bucket = aws_s3_bucket.frontend.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_policy" "allow_public_read" {
  bucket = aws_s3_bucket.frontend.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicReadGetObject"
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.frontend.arn}/*"
      },
    ]
  })

  depends_on = [aws_s3_bucket_public_access_block.frontend]
}

# DynamoDB table for caching (free tier)
resource "aws_dynamodb_table" "cache_table" {
  name           = "${var.project_name}-cache"
  billing_mode   = "PAY_PER_REQUEST" # Free tier eligible
  hash_key       = "query_hash"
  attribute {
    name = "query_hash"
    type = "S"
  }

  ttl {
    attribute_name = "expire_time"
    enabled        = true
  }

  tags = {
    Name        = "Finance RAG Cache"
    Environment = var.environment
  }
}

# Lambda function (free tier)
resource "aws_lambda_function" "api_function" {
  function_name = "${var.project_name}-api"
  role          = aws_iam_role.lambda_exec.arn
  handler       = "index.handler"
  runtime       = "python3.10"
  timeout       = 30
  memory_size   = 256 # Low memory to stay in free tier

  # Using dummy code - actual deployment would be done via serverless.yml
  filename      = "${path.module}/dummy_lambda.zip"
  
  environment {
    variables = {
      STAGE = var.environment
      DATABASE_URL = var.database_url
      VECTOR_DB_PATH = var.vector_db_path
    }
  }
}

# API Gateway (free tier)
resource "aws_apigatewayv2_api" "api" {
  name          = "${var.project_name}-http-api"
  protocol_type = "HTTP"
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_headers = ["content-type", "authorization"]
    max_age       = 300
  }
}

resource "aws_apigatewayv2_stage" "api" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = var.environment
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gw.arn

    format = jsonencode({
      requestId               = "$context.requestId"
      sourceIp                = "$context.identity.sourceIp"
      requestTime             = "$context.requestTime"
      protocol                = "$context.protocol"
      httpMethod              = "$context.httpMethod"
      resourcePath            = "$context.resourcePath"
      routeKey                = "$context.routeKey"
      status                  = "$context.status"
      responseLength          = "$context.responseLength"
      integrationErrorMessage = "$context.integrationErrorMessage"
      }
    )
  }
}

resource "aws_apigatewayv2_integration" "api_lambda" {
  api_id             = aws_apigatewayv2_api.api.id
  integration_uri    = aws_lambda_function.api_function.invoke_arn
  integration_type   = "AWS_PROXY"
  integration_method = "POST"
}

resource "aws_apigatewayv2_route" "api_route" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "POST /query"
  target    = "integrations/${aws_apigatewayv2_integration.api_lambda.id}"
}

resource "aws_cloudwatch_log_group" "api_gw" {
  name              = "/aws/api_gw/${aws_apigatewayv2_api.api.name}"
  retention_in_days = 7 # Minimal retention to save space in free tier
}

# Lambda permissions
resource "aws_lambda_permission" "api_gw" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api_function.function_name
  principal     = "apigateway.amazonaws.com"

  source_arn = "${aws_apigatewayv2_api.api.execution_arn}/*/*"
}

# IAM role with minimal permissions 
resource "aws_iam_role" "lambda_exec" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_policy" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Custom policy for DynamoDB access
resource "aws_iam_policy" "dynamodb_access" {
  name        = "${var.project_name}-dynamodb-policy"
  description = "IAM policy for DynamoDB access"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:DeleteItem",
          "dynamodb:Query",
        ]
        Effect   = "Allow"
        Resource = aws_dynamodb_table.cache_table.arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "dynamodb_policy_attachment" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = aws_iam_policy.dynamodb_access.arn
} 