#!/bin/bash
# Script to run the Finance RAG system evaluation

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Create output directory
mkdir -p evaluation_results

# Install dependencies if not already installed
if ! pip show psutil > /dev/null 2>&1; then
    echo "Installing required dependencies..."
    pip install psutil
fi

# Run the evaluation script
echo "Starting evaluation..."
python src/evaluation/run_evaluation.py "$@"

# Check if the evaluation was successful
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully."
    echo "Results saved to evaluation_results directory."
else
    echo "Evaluation failed. Check the logs for details."
    exit 1
fi

echo "To view the results, check the JSON files in the evaluation_results directory." 