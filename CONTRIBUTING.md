# Contributing to Finance RAG Application

Thank you for your interest in contributing to the Finance RAG Application! This document provides guidelines and instructions for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub by clicking the Fork button.
2. **Clone your fork** to your local machine:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Personal_Finance.git
   cd Personal_Finance
   ```
3. **Set up the development environment**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Install frontend dependencies
   cd app
   npm install
   cd ..
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Backend Development

1. Make changes to the Python code in the `src/` directory.
2. Run tests to ensure your changes don't break existing functionality:
   ```bash
   python -m pytest tests/
   ```
3. Run linting checks:
   ```bash
   flake8 src/ tests/
   ```

### Frontend Development

1. Make changes to the React code in the `app/` directory.
2. Run the development server:
   ```bash
   cd app
   npm run dev
   ```
3. Check for linting issues:
   ```bash
   npm run lint
   ```

## Local Testing with Docker

You can test the entire application locally using Docker:

```bash
docker-compose up
```

This will start both the backend and frontend services.

## Pull Request Process

1. **Update your fork** with the latest changes from the main repository:
   ```bash
   git remote add upstream https://github.com/jaineet17/Personal_Finance.git
   git fetch upstream
   git merge upstream/main
   ```

2. **Push your changes** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a pull request** from your branch to the main repository's `dev` branch.

4. **Describe your changes** in the pull request:
   - What changes you've made
   - Why you've made these changes
   - Any relevant issue numbers

5. **Wait for review**. Maintainers will review your PR and may request changes.

## Code Style Guidelines

### Python

- Follow PEP 8 guidelines
- Use type hints wherever possible
- Write docstrings for functions and classes
- Keep functions small and focused
- Use meaningful variable and function names

### JavaScript/TypeScript

- Follow the ESLint configuration in the project
- Use functional components and hooks for React
- Use TypeScript interfaces for type definitions
- Keep components small and focused
- Use meaningful variable and function names

## Commit Message Guidelines

- Use clear and meaningful commit messages
- Start with a verb in the imperative form (e.g., "Add", "Fix", "Update")
- Keep the first line under 72 characters
- Reference issue numbers when applicable

Example:
```
Add transaction categorization feature

- Add category detection logic
- Create category matching algorithm
- Update database schema to support categories
- Add tests for new functionality

Fixes #42
```

## Documentation

- Update documentation when making changes to functionality
- Document new features or significant changes
- Update README.md if necessary
- Add JSDoc comments to JavaScript/TypeScript functions

## Testing

- Write tests for new functionality
- Ensure existing tests pass with your changes
- Include both unit and integration tests when applicable
- For frontend changes, include visual tests when possible

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

- The CI workflow runs on every push and pull request
- The CD-Dev workflow deploys to the development environment
- The CD-Prod workflow deploys to production on releases

Ensure your changes pass all CI checks before requesting a review.

## Questions and Support

If you have questions or need support:

1. Check existing issues to see if your question has been answered
2. Create a new issue with the "question" label
3. Provide as much context as possible

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License. 