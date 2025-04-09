#!/bin/bash

# Path to the input file containing experiment names
INPUT_FILE="filtered_experiments.txt"

# Check if the file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File $INPUT_FILE not found."
    exit 1
fi

# Initialize a counter
count=0

# Read the file line by line
while IFS= read -r experiment_name; do
    # Skip empty lines
    if [ -z "$experiment_name" ]; then
        continue
    fi
    
    echo "Processing experiment: $experiment_name"
    
    # Run the command with the experiment name as an argument
    uv run python run_bertscore_evaluation.py "$experiment_name"
    
    # Increment counter
    ((count++))
    
    echo "Completed $count experiments"
    echo "-----------------------------------"
    
done < "$INPUT_FILE"

echo "All experiments processed. Total count: $count"