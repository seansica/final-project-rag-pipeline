#!/bin/bash
# Script to run GEval evaluation on all experiment runs in a file

# Exit on any error
set -e

# Path to the evaluation script
EVAL_SCRIPT="./run_geval_evaluation.py"

# Check if a file is provided
if [ $# -ne 0 ]; then
    echo "Using provided experiment names from file: $1"
    # Read experiment names from the file
    mapfile -t EXPERIMENTS < "$1"
else
    echo "No input file provided. Please provide a file with experiment names."
    echo "Usage: $0 experiment_list.txt"
    exit 1
fi

# Run evaluations
total=${#EXPERIMENTS[@]}
count=0

for exp in "${EXPERIMENTS[@]}"; do
    count=$((count + 1))
    echo "[$count/$total] Evaluating experiment: $exp"
    
    # Run the evaluation script with uv
    uv run python "$EVAL_SCRIPT" "$exp" || {
        echo "Warning: Evaluation failed for experiment: $exp"
        # Continue with the next experiment
    }
    
    # Add a small delay to avoid API rate limits
    sleep 5
    
    echo "-----------------------------------"
done

echo "All evaluations completed!"