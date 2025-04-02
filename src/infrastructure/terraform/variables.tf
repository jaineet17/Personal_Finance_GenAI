variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name of the project used for resource naming"
  type        = string
  default     = "finance-rag"
}

variable "environment" {
  description = "Environment (dev/prod)"
  type        = string
  default     = "dev"
}

variable "frontend_bucket_name" {
  description = "Name of the S3 bucket for static website hosting"
  type        = string
  default     = "finance-rag-frontend"
}

variable "database_url" {
  description = "Database connection string (MongoDB Atlas free tier or sqlite)"
  type        = string
  default     = "sqlite:///./finance.db"
}

variable "vector_db_path" {
  description = "Path to the vector database"
  type        = string
  default     = "./vector_db"
}

variable "enable_ollama_connection" {
  description = "Whether to enable connection to local Ollama server"
  type        = bool
  default     = false
}

variable "ollama_api_url" {
  description = "URL for Ollama API (if enable_ollama_connection=true)"
  type        = string
  default     = "http://localhost:11434"
}

variable "use_mongo_atlas" {
  description = "Whether to use MongoDB Atlas as the database"
  type        = bool
  default     = false
}

variable "mongo_atlas_connection_string" {
  description = "MongoDB Atlas connection string (if use_mongo_atlas=true)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "lambda_memory_size" {
  description = "Memory allocation for Lambda functions (MB)"
  type        = number
  default     = 256
}

variable "lambda_timeout" {
  description = "Lambda function timeout (seconds)"
  type        = number
  default     = 30
}

variable "dynamodb_ttl_days" {
  description = "Number of days before DynamoDB records expire"
  type        = number
  default     = 30
}

variable "log_retention_days" {
  description = "Number of days to retain CloudWatch logs"
  type        = number
  default     = 7
} 