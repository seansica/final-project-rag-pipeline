# Running Missing Experiments Guide

This guide explains how to use the `run_single_missing_experiment.py` script to recover from incomplete experiment runs by running a single missing experiment and appending it to existing results.

## Overview

During batch execution of RAG experiments, sometimes individual experiments may fail or get interrupted. The `run_single_missing_experiment.py` script allows you to:

1. Run a specific experiment configuration that was missed
2. Append the results to an existing results.json file
3. Complete your experiment suite without rerunning all experiments

## Basic Usage

```bash
python run_single_missing_experiment.py --results_file /path/to/results.json
```

By default, this will:
- Run an experiment with the missing configuration (default values focused on marketing team with all-MiniLM-L6-v2 embedding model)
- Append the results to the specified results.json file
- Use default values for chunking and retrieval parameters

## Command-Line Parameters

| Parameter | Description |
|-----------|-------------|
| `--results_file` | Required. Path to the existing results.json file |
| `--rag_type` | RAG type (default: cohere) |
| `--team_type` | Team type (default: marketing) |
| `--embedding_model` | Embedding model (default: all-MiniLM-L6-v2) |
| `--chunk_size` | Chunk size (default: 128) |
| `--chunk_overlap` | Chunk overlap (default: 0) |
| `--top_k` | Top k (default: 4) |
| `--limit` | Question limit (default: 78) |
| `--no_ragas` | Use standard evaluations instead of RAGAS |

## Example Scenarios

### Running a Specific Missing Experiment

If you know exactly which experiment is missing:

```bash
python run_single_missing_experiment.py \
  --results_file results/phase1_ragas_20250404_232654/results.json \
  --team_type marketing \
  --embedding_model all-MiniLM-L6-v2
```

### Testing with a Small Question Set

To run faster with fewer questions:

```bash
python run_single_missing_experiment.py \
  --results_file results/phase1_ragas_20250404_232654/results.json \
  --limit 10
```

### Switching Evaluation Types

To use standard evaluations instead of RAGAS:

```bash
python run_single_missing_experiment.py \
  --results_file results/phase1_standard_20250404_201205/results.json \
  --no_ragas
```

## Identifying Missing Experiments

To identify which experiments are missing from your results:

1. Compare the experiment plan with the actual results:
   ```bash
   # Count experiments in plan
   grep -o "embedding_model" results/phase1_*/experiment_plan.json | wc -l
   
   # Count successful experiments in results
   grep -o "success\": true" results/phase1_*/results.json | wc -l
   ```

2. Find specific missing configurations:
   ```bash
   # Check for a specific experiment ID in results
   grep "ragas-rag-cohere-marketing-emb-all-MiniLM-L6-v2" results/phase1_*/results.json
   ```

## How It Works

The script:
1. Loads the existing results.json file
2. Creates an ExperimentConfig with the specified parameters
3. Generates a unique experiment ID based on the configuration
4. Checks if the experiment already exists in the results
5. Runs the evaluation using the same core function as the main script
6. Appends the new result to the existing results
7. Writes the updated results back to the file

## Troubleshooting

### Experiment Already Exists

If the script reports that the experiment already exists:
- Double-check your parameters to ensure you're targeting the correct missing experiment
- Check the results file to confirm which experiments have been completed

### API and Resource Issues

If the script fails due to API or resource issues:
- Try a smaller question limit to reduce resource usage
- Ensure API keys are properly configured in environment variables
- Check for CUDA memory issues if using GPU-based models