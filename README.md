# Finance LLM Application with RAG

A powerful financial assistant application that uses Retrieval-Augmented Generation (RAG) to provide intelligent insights and answers about your personal finances.

## Features

- **Natural Language Queries**: Ask questions about your finances in plain English
- **Transaction Analysis**: Analyze spending patterns across categories and time periods
- **Smart Transaction Categorization**: AI-powered categorization of financial transactions
- **Budget Recommendations**: Get personalized budget recommendations based on your spending history
- **Financial Insights**: Discover spending trends and actionable insights
- **Multiple LLM Support**: Works with OpenAI, Anthropic, HuggingFace, and local models
- **Real Financial Data Testing**: Comprehensive test suite that works with your actual financial data
- **Local Deployment**: Easy to deploy and run on your local machine
- **Responsive Web Interface**: Modern React frontend for easy interaction with the financial assistant
- **Comprehensive Testing**: Automated testing through GitHub Actions CI

## Local Deployment

### Prerequisites

- Python 3.10+
- Node.js 16+
- Git

### Installation

1. Clone the repository
```bash
git clone https://github.com/jaineet17/Personal_Finance_GenAI.git
cd Personal_Finance_GenAI
```

2. Install Python dependencies
```bash
pip install -r requirements.txt
```

3. Install frontend dependencies
```bash
cd frontend
npm install
cd ..
```

### Running the Application

1. Start the backend API
```bash
python src/app.py
```

2. In a new terminal, start the frontend
```bash
cd frontend
npm run dev
```

3. Access the application at http://localhost:5173

## Testing

Run the comprehensive test suite:

```bash
python -m tests.run_tests
```

Or run specific test modules:

```bash
python -m tests.run_tests -m test_data_processing
```

## Project Structure

The project follows a clean architecture with the following components:

1. **Data Processing**: Handles ingestion and standardization of financial data
2. **Vector Database**: Stores embeddings for semantic search
3. **RAG Engine**: Combines retrieval with LLM generation
4. **API Layer**: Exposes functionality through FastAPI
5. **React Frontend**: Provides user interface

## Documentation

Detailed documentation is available in the `docs/` directory:
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- [Cloud Security](docs/CLOUD_SECURITY.md) 
- [Frontend Deployment](docs/FRONTEND_DEPLOYMENT.md)
- [Monitoring](docs/MONITORING.md)
- [Phase 6 Completion](docs/PHASE_6_COMPLETION.md)

## License

This project is licensed under the MIT License.

## Acknowledgements

- [SentenceTransformers](https://www.sbert.net/) for text embeddings
- [ChromaDB](https://github.com/chroma-core/chroma) for vector search
- [OpenAI](https://openai.com/) and [Anthropic](https://www.anthropic.com/) for LLM APIs 

## Project Status

All phases of the project have been completed:

- Phase 4: API and User Interface Development
- Phase 5: Evaluation and Optimization
- Phase 6: Implementation and Documentation

The system features:
- Complete local development environment
- Robust testing framework
- Well-documented code and architecture
- Responsive frontend interface
- Comprehensive RAG implementation
- GitHub Actions CI for code quality assurance

## Getting Started

To start using the Finance RAG application:

1. Clone this repository
2. Follow the setup instructions in the installation section above
3. Import your financial data using the included importers
4. Access the web interface or use the CLI to interact with your financial data

For development purposes, you can also run the application locally using Docker:

```bash
docker-compose up
```

This will start both the backend API server and frontend development server 