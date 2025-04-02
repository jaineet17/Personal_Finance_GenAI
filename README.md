# Finance LLM Application with RAG

A powerful financial assistant application that uses Retrieval-Augmented Generation (RAG) to provide intelligent insights and answers about your personal finances. The application is fully deployable to AWS cloud services for scalable and secure access.

## Features

- **Natural Language Queries**: Ask questions about your finances in plain English
- **Transaction Analysis**: Analyze spending patterns across categories and time periods
- **Smart Transaction Categorization**: AI-powered categorization of financial transactions
- **Budget Recommendations**: Get personalized budget recommendations based on your spending history
- **Financial Insights**: Discover spending trends and actionable insights
- **Multiple LLM Support**: Works with OpenAI, Anthropic, HuggingFace, and local models
- **Real Financial Data Testing**: Comprehensive test suite that works with your actual financial data
- **Cloud Deployment**: Fully deployable to AWS (ECS/Fargate, Lambda) and MongoDB Atlas
- **Responsive Web Interface**: Modern React frontend for easy interaction with the financial assistant
- **CI/CD Pipeline**: Automated testing and deployment workflows with GitHub Actions

## Project Structure

```
finance-llm-app/
├── .github/               # GitHub Actions workflows for CI/CD
│   └── workflows/         # CI/CD workflow configuration files
├── app/                   # React frontend application
│   ├── components/        # React components
│   ├── pages/             # React pages
│   ├── styles/            # CSS and styling
│   └── utils/             # Frontend utilities
├── data/                  # Data directory
│   ├── raw/               # Raw data files
│   ├── processed/         # Processed data
│   └── vectors/           # Vector embeddings storage
├── deployment/            # Deployment scripts and configurations
│   ├── aws/               # AWS deployment configurations
│   └── docker/            # Docker configurations
├── docs/                  # Documentation
│   ├── DEPLOYMENT_GUIDE.md  # Deployment documentation
│   ├── CLOUD_SECURITY.md    # Security implementation details
│   └── CLOUD_DATA_MANAGEMENT.md # Data management strategies
├── src/                   # Source code
│   ├── api/               # API endpoints
│   ├── cli/               # Command line interface
│   ├── data_processing/   # Data processing scripts
│   ├── db/                # Database interactions
│   ├── embedding/         # Embedding generation and vector store
│   ├── importers/         # Data importers for different sources
│   ├── infrastructure/    # IaC configurations (Serverless, CDK)
│   ├── llm/               # LLM client interfaces
│   ├── models/            # Data models
│   ├── processing/        # Data processing utilities
│   ├── prompts/           # LLM prompts and templates
│   ├── rag/               # RAG system components
│   ├── retrieval/         # Retrieval components
│   └── utils/             # Utility functions
├── tests/                 # Test cases
│   ├── test_integration.py  # Integration tests with real financial data
│   └── test_queries.json    # Test queries for validation
├── test_output/           # Test results output directory
├── Dockerfile             # Docker container definition
├── deploy_fargate.sh      # ECS/Fargate deployment script
├── .env                   # Environment variables
└── requirements.txt       # Dependencies
```

## Detailed Component Documentation

### Data Processing Pipeline

The data processing pipeline handles the ingestion, cleaning, and standardization of financial transaction data.

**Key Components:**
- **DataProcessor (`src/data_processing/processor.py`)**: The main class for processing raw financial data files, handling different file formats, and standardizing column names.
  - Detects file types and applies appropriate processing logic
  - Standardizes date formats, transaction descriptions, and amounts
  - Assigns unique IDs to transactions
  - Outputs processed CSV files ready for database import

- **process_data.py (`src/data_processing/process_data.py`)**: A command-line script that orchestrates the entire data processing workflow:
  - Processes both regular CSV files and specialized formats (Chase, American Express)
  - Handles case sensitivity in file extensions (`.csv` vs `.CSV`)
  - Properly escapes special characters in transaction descriptions
  - Creates a clean SQLite database with standardized schema
  - Extracts month information for time-based aggregation

### Database Layer

The database layer provides persistent storage and efficient querying of financial transaction data.

**Key Components:**
- **Database (`src/db/database.py`)**: Handles all database operations including:
  - Table creation for transactions, categories, accounts, and monthly aggregates
  - Transaction import with validation and deduplication
  - Flexible querying with filters for date ranges, categories, accounts, and amounts
  - Monthly spending aggregations by category and account
  - Support for SQLite with a standardized schema

### Embedding and Vector Storage

The embedding system converts financial transactions and queries into vector representations for semantic search.

**Key Components:**
- **VectorStore (`src/embedding/vector_store.py`)**: Manages the vector database backed by ChromaDB:
  - Creates and maintains collections for transactions, categories, and time periods
  - Handles addition, deletion, and updating of vector embeddings
  - Provides methods for similarity search across collections
  - Supports hybrid search (combination of semantic and metadata filtering)
  
- **Embedding Generation**: Uses the SentenceTransformers library to create high-quality embeddings:
  - Transforms transaction descriptions into vector embeddings
  - Creates rich embeddings that capture semantic meaning of financial terms
  - Ensures consistent vector dimensions and normalization
  - Supports multiple embedding models with different quality/performance tradeoffs

### Retrieval System

The retrieval system finds and ranks relevant financial information based on user queries.

**Key Components:**
- **FinanceRetrieval (`src/retrieval/retrieval_system.py`)**: Coordinates the retrieval process:
  - Parses natural language queries to extract key information
  - Determines appropriate retrieval strategy based on query type
  - Executes hybrid search combining vector similarity and metadata filters
  - Ranks and scores results based on relevance
  - Returns structured information ready for LLM consumption

- **Query Understanding**: Analyzes natural language queries to extract:
  - Time periods and date ranges
  - Categories and merchants
  - Financial metrics (spending, saving, etc.)
  - Comparison operations
  - Action intents (analyze, categorize, forecast)

### RAG System

The RAG (Retrieval-Augmented Generation) system combines retrieved data with an LLM to generate accurate financial insights.

**Key Components:**
- **FinanceRAG (`src/rag/finance_rag.py`)**: The main orchestrator of the RAG process:
  - Connects to various LLM providers (OpenAI, Anthropic, Hugging Face, Ollama)
  - Creates prompts with retrieved financial data
  - Manages context length constraints
  - Handles LLM errors and fallbacks
  - Post-processes LLM responses for consistency

- **Prompt Engineering (`src/prompts/`)**: Creates effective prompts for the LLM:
  - Includes system prompts that define the assistant's persona
  - Provides examples and few-shot learning demonstrations
  - Formats retrieved financial data for maximum context utilization
  - Guides the LLM to provide structured and accurate responses

### Testing Framework

The testing framework ensures system reliability and validates performance against real financial data.

**Key Components:**
- **Integration Tests (`tests/test_integration.py`)**: End-to-end tests for the entire system:
  - Tests each component individually and in combination
  - Validates retrieval accuracy and LLM response quality
  - Exports comprehensive test results for analysis
  - Adapts to real financial data with flexible assertions

- **Test Queries (`tests/test_queries.json`)**: A collection of test queries for validation:
  - Basic, intermediate, and advanced financial queries
  - Real-data specific test cases
  - Categorization examples
  - Expected properties and retrieval counts

### Command-Line Interface

The CLI provides a user-friendly interface to interact with the financial RAG system.

**Key Components:**
- **RAG CLI (`src/cli/rag_cli.py`)**: Command-line interface with multiple modes:
  - Interactive query mode with conversation history
  - Direct query mode for quick answers
  - Analysis mode for deeper financial insights
  - Categorization mode for transaction management
  - Similar transaction search mode

## Implementation Details

### Data Flow

The typical data flow through the system is:

1. **Data Ingestion**: Raw financial data is processed via `process_data.py`
2. **Data Storage**: Processed data is stored in SQLite database
3. **Vector Embedding**: Transaction data is embedded and stored in ChromaDB
4. **Query Processing**: User query is parsed and embedded
5. **Retrieval**: Relevant transactions and categories are retrieved
6. **Prompt Creation**: Retrieved data is formatted into a prompt
7. **LLM Generation**: Prompt is sent to LLM for response generation
8. **Response Delivery**: LLM response is returned to the user

### Transaction Processing

Each transaction follows this processing pipeline:

1. **Raw Data Extraction**: Read from source files (CSV, OFX, etc.)
2. **Column Standardization**: Map source columns to standard schema
3. **Data Cleaning**: Handle special characters, fix formatting issues
4. **ID Assignment**: Generate unique IDs for deduplication
5. **Category Assignment**: Apply initial categorization
6. **Database Storage**: Insert into SQLite database
7. **Vector Embedding**: Create and store vector representations
8. **Aggregation**: Update monthly spending aggregates

### Query Processing

When a user submits a natural language query, the system:

1. **Parse Query**: Extract key entities, time periods, and intents
2. **Construct Search Strategy**: Determine appropriate search parameters
3. **Vector Search**: Find semantically similar transactions
4. **Metadata Filtering**: Apply filters for dates, categories, amounts
5. **Result Ranking**: Score and rank retrieval results
6. **Context Building**: Format retrieved data for LLM prompt
7. **LLM Querying**: Send prompt to LLM for response generation

## Data Processing

The system includes a dedicated data processing pipeline for preparing your financial data:

```bash
# Process raw financial data
python src/data_processing/process_data.py
```

This script:
- Processes raw CSV files from various financial institutions
- Handles different file formats (including Chase and American Express)
- Standardizes transaction data format
- Creates a SQLite database with your financial transactions
- Prepares data for vector embedding

Supported raw data formats:
- Standard CSV files with transaction data
- Chase Bank CSV files (credit cards and checking accounts)
- American Express transaction downloads

## Testing with Real Financial Data

The application includes a comprehensive test suite that can run against your actual financial data:

```bash
# Run tests with real financial data
mkdir -p test_output && export DATA_DIR=data && python -m tests.run_tests
```

The test suite:
- Validates the RAG system using real financial queries
- Tests hybrid search functionality 
- Verifies embedding generation and similarity calculations
- Exports detailed test results to JSON files in the test_output directory
- Adapts to your specific transaction data

Test results are saved in the `test_output` directory for analysis and fine-tuning of the system.

## Data Import

The system supports importing financial data from various sources:

- CSV files from banks and financial institutions
- QFX/OFX files
- JSON transaction data
- SQL databases

For example, to import transactions from a CSV file:

```bash
python src/importers/csv_importer.py --file your_transactions.csv
```

## Future Development

For details on future development plans, please refer to the following documentation:

- [Implementation Plan for Phases 4-6](docs/implementation_plan_phases_4-6.md) - Detailed steps for API development, frontend implementation, evaluation, optimization, and cloud deployment
- [Cloud Deployment Readiness Checklist](docs/deployment_readiness_checklist.md) - Comprehensive checklist to ensure the system is ready for cloud deployment

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [SentenceTransformers](https://www.sbert.net/) for text embeddings
- [ChromaDB](https://github.com/chroma-core/chroma) for vector search
- [OpenAI](https://openai.com/) and [Anthropic](https://www.anthropic.com/) for LLM APIs 

## Cloud Deployment

The application can be deployed to AWS using either ECS/Fargate or Serverless Lambda:

### ECS/Fargate Deployment

Deploy the application as a containerized service:

```bash
# Set required environment variables
export HUGGINGFACE_API_KEY=your_key_here
export MONGODB_URI=your_mongo_connection_string

# Run the deployment script
./deploy_fargate.sh
```

The deployment script:
- Creates an ECR repository for the Docker image
- Builds and pushes the Docker container
- Creates an ECS cluster
- Sets up IAM roles and security groups
- Configures load balancing
- Deploys the application as a Fargate service

### Serverless Lambda Deployment

Deploy as a serverless function:

```bash
cd src/infrastructure
serverless deploy
```

This deployment method:
- Creates API Gateway endpoints
- Sets up Lambda functions
- Configures proper IAM permissions
- Manages environment variables securely
- Enables easy scaling for varying workloads

## Frontend Application

The project includes a responsive React-based frontend:

```bash
# Install dependencies
cd app
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

Features of the frontend:
- Chat-like interface for querying financial data
- Transaction visualization dashboard
- Category management tools
- Budget tracking and recommendations
- Mobile-responsive design

## CI/CD Pipeline

The project includes GitHub Actions workflows for continuous integration and deployment:

- **CI Workflow**: Runs on every push and pull request
  - Lints and validates code
  - Runs unit and integration tests
  - Validates Docker builds

- **CD-Dev Workflow**: Deploys to development environment on pushes to dev branch
  - Builds and deploys backend to AWS
  - Builds and deploys frontend to Vercel/Netlify
  - Updates environment variables

- **CD-Prod Workflow**: Deploys to production on release publication
  - Builds and deploys backend to AWS production environment
  - Builds and deploys frontend to production
  - Creates database backups

## Security and Data Management

The project implements comprehensive security and data management strategies:

- **Security**: See `docs/CLOUD_SECURITY.md` for details on:
  - Authentication and authorization
  - Data encryption
  - API security
  - Network security
  - Monitoring and alerting

- **Data Management**: See `docs/CLOUD_DATA_MANAGEMENT.md` for details on:
  - MongoDB Atlas optimization
  - Data retention policies
  - Backup strategies
  - Performance optimization

## Documentation

Complete documentation is available in the `docs/` directory:

- **Deployment Guide**: `docs/DEPLOYMENT_GUIDE.md` - Step-by-step deployment instructions
- **Cloud Security**: `docs/CLOUD_SECURITY.md` - Security implementation details
- **Data Management**: `docs/CLOUD_DATA_MANAGEMENT.md` - Data management strategies
- **Implementation Plan**: `docs/implementation_plan_phases_4-6.md` - Project phases and milestones
- **Frontend Deployment**: `docs/FRONTEND_DEPLOYMENT.md` - Frontend build and deployment guide
- **Monitoring**: `docs/MONITORING.md` - Monitoring configuration and strategies

## Project Status

All phases of the project have been completed:

- Phase 4: API and User Interface Development
- Phase 5: Evaluation and Optimization
- Phase 6: Cloud Deployment Implementation

The system is now fully deployed to AWS with:
- Containerized backend in AWS ECS/Fargate
- Serverless API options using AWS Lambda
- Static frontend hosted on Vercel/Netlify
- Automated CI/CD pipelines with GitHub Actions
- Comprehensive monitoring with CloudWatch
- Security implementation based on best practices
- Cost-optimized architecture within AWS free tier limits

## Getting Started

To start using the Finance RAG application:

1. Clone this repository
2. Follow the setup instructions in the [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
3. Import your financial data using the included importers
4. Access the web interface or use the CLI to interact with your financial data

For development purposes, you can also run the application locally using Docker:

```bash
docker-compose up
```

This will start both the backend API server and frontend development server 