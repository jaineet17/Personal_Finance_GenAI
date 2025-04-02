#!/bin/bash
# Script to run the complete Finance RAG system evaluation and optimization pipeline

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Create output directory
mkdir -p evaluation_results

# Install dependencies if not already installed
if ! pip show psutil matplotlib pandas numpy > /dev/null 2>&1; then
    echo "Installing required dependencies..."
    pip install psutil matplotlib pandas numpy
fi

# Define default and custom arguments
DEFAULT_ARGS="--output-dir evaluation_results --query-sets basic_queries,intermediate_queries"
CUSTOM_ARGS="$@"

# Run the full evaluation pipeline
echo "Starting the full evaluation and optimization pipeline..."
python src/evaluation/run_full_evaluation.py $DEFAULT_ARGS $CUSTOM_ARGS

# Check if the evaluation was successful
if [ $? -eq 0 ]; then
    echo "Evaluation and optimization pipeline completed successfully."
    echo "Results saved to evaluation_results directory."
    
    # If a dashboard was generated, try to open it
    if [ -f "evaluation_results/dashboard.html" ]; then
        echo "Dashboard available at: evaluation_results/dashboard.html"
        
        # Try to open the dashboard automatically if on a supported system
        if [ "$(uname)" == "Darwin" ]; then
            # MacOS
            open evaluation_results/dashboard.html
        elif [ "$(uname)" == "Linux" ] && command -v xdg-open > /dev/null; then
            # Linux with xdg-open
            xdg-open evaluation_results/dashboard.html
        elif command -v start > /dev/null; then
            # Windows
            start evaluation_results/dashboard.html
        else
            echo "Please open the dashboard manually in your web browser."
        fi
    fi
else
    echo "Evaluation and optimization pipeline failed. Check the logs for details."
    exit 1
fi 