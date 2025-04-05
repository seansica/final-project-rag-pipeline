# Phased RAG Evaluation Guide

This guide explains how to use the `run_phased_evaluations.py` script to systematically evaluate different configurations of RAG (Retrieval-Augmented Generation) systems in a phased approach.

## Overview

The script runs a series of experiments across three optimization phases:

1. **Phase 1: Embedding Model Selection** - Compare different embedding models
2. **Phase 2: Chunking Strategy Selection** - Optimize chunk size and overlap
3. **Phase 3: Retriever Method Selection** - Compare different retrieval methods and parameters

Each phase builds on the results of the previous phase, allowing you to systematically discover the optimal RAG configuration for your use case.

## Prerequisites

- Python environment with required packages
- LangSmith API credentials configured in environment variables
- Cohere API credentials configured in environment variables

## Basic Usage

### Phase 1: Embedding Model Selection

```bash
python run_phased_evaluations.py --phase 1 --max_parallel 2 --ragas
```

This will:
- Run experiments with different embedding models
- Use the default chunk_size (128), chunk_overlap (0), and top_k (4)
- Use Ragas metrics for evaluation if `--ragas` flag is included

### Phase 2: Chunking Strategy Selection

After analyzing Phase 1 results and identifying the best embedding model:

```bash
python run_phased_evaluations.py --phase 2 --embedding_model "multi-qa-mpnet-base-dot-v1" --max_parallel 2 --ragas
```

This will:
- Run experiments with different chunk sizes and overlap values
- Use the specified embedding model from Phase 1
- Keep using the same default retriever_type (similarity)

### Phase 3: Retriever Method Selection

After analyzing Phase 2 results:

```bash
python run_phased_evaluations.py --phase 3 --embedding_model "multi-qa-mpnet-base-dot-v1" --chunk_size 256 --chunk_overlap 50 --max_parallel 2 --ragas
```

This will:
- Run experiments with different retriever methods (similarity, similarity with threshold, MMR, multi-query)
- Use the optimal embedding model, chunk size and overlap from previous phases

## Command-Line Parameters

| Parameter | Description |
|-----------|-------------|
| `--phase` | Required. Experiment phase to run (1, 2, or 3) |
| `--max_parallel` | Maximum number of parallel experiments (default: 2) |
| `--output_dir` | Directory to save results (default: auto-generated timestamp-based name) |
| `--embedding_model` | Best embedding model from Phase 1 (required for Phases 2 and 3) |
| `--chunk_size` | Best chunk size from Phase 2 (required for Phase 3) |
| `--chunk_overlap` | Best chunk overlap from Phase 2 (required for Phase 3) |
| `--ragas` | Use Ragas metrics instead of standard evaluations |
| `--limit` | Limit the number of questions used in evaluation (default: 78) |

## Evaluation Types

The script supports two types of evaluations:

### Standard Evaluations (default)

- correctness
- groundedness
- relevance
- retrieval_relevance

### Ragas Evaluations (with `--ragas` flag)

- ragas_answer_accuracy
- ragas_context_relevance  
- ragas_faithfulness
- ragas_response_relevancy

## Performance Optimization

### Limiting Dataset Size

For faster experimentation, you can use a subset of the validation dataset:

```bash
python run_phased_evaluations.py --phase 1 --max_parallel 2 --ragas --limit 10
```

This will run the evaluation using only 10 questions instead of the full dataset.

### Parallel Execution

The `--max_parallel` parameter controls how many experiments can run concurrently. Adjust based on your machine's capabilities. For Cohere API-based experiments, 2-3 is generally optimal. Note that Mistral experiments always run sequentially due to GPU memory constraints.

## Output and Results

The script creates a directory in the `results/` folder with:

- `experiment_plan.json`: Details of all experiments to be run
- `results.json`: Results of all completed experiments, including metrics

Example output directory: `results/phase1_ragas_20250404_232654/`

### Metrics in Results

The `results.json` file contains detailed metrics for each experiment:

- Experiment configuration details
- Success status and execution time
- Feedback metrics (e.g., answer accuracy, context relevance)
- Text comparison statistics (character length, word count)

## Analysis

After running experiments, you can analyze the results using the `analyze_top_performers.py` script:

```bash
python analyze_top_performers.py /path/to/results.json --output_dir ./analysis --visualize
```

This will:
- Identify the top-performing configurations for each metric
- Calculate overall rankings across all metrics
- Generate visualizations to help interpret the results

## Troubleshooting

### Memory Issues

If you encounter CUDA memory errors:
- Reduce the `--limit` parameter to process fewer questions
- Reduce `--max_parallel` to run fewer experiments concurrently
- Consider using CPU-based embedding models

### API Rate Limits

If you hit API rate limits:
- Reduce `--max_parallel` to reduce concurrent API calls
- Add delays between API calls in the code if needed

## Example Workflow

1. Run Phase 1 to find the best embedding model:
   ```bash
   python run_phased_evaluations.py --phase 1 --max_parallel 2 --ragas --limit 20
   ```

2. Analyze results to identify the best embedding model:
   ```bash
   python analyze_top_performers.py results/phase1_ragas_*/results.json
   ```

3. Run Phase 2 with the best embedding model:
   ```bash
   python run_phased_evaluations.py --phase 2 --embedding_model "all-mpnet-base-v2" --max_parallel 2 --ragas --limit 20
   ```

4. Analyze Phase 2 results:
   ```bash
   python analyze_top_performers.py results/phase2_ragas_*/results.json
   ```

5. Run Phase 3 with the optimal configuration:
   ```bash
   python run_phased_evaluations.py --phase 3 --embedding_model "all-mpnet-base-v2" --chunk_size 512 --chunk_overlap 50 --max_parallel 2 --ragas --limit 20
   ```

6. Final analysis of Phase 3 to determine the best overall RAG configuration:
   ```bash
   python analyze_top_performers.py results/phase3_ragas_*/results.json --output_dir ./final_analysis --visualize
   ```