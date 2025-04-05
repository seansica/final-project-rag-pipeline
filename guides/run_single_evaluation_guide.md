# Single RAG Evaluation Guide

This guide explains how to use the `run_single_evaluation.py` script to evaluate a specific RAG configuration without running a full phased experiment suite.

## Overview

The `run_single_evaluation.py` script allows you to evaluate a single RAG configuration with detailed control over all parameters, including:

- RAG system type (Cohere or Mistral)
- Team type (Engineering or Marketing)
- Embedding model
- Chunk size and overlap
- Retriever type and parameters
- Evaluation metrics (standard or RAGAS)

This is useful for:
- Quick testing of a specific configuration
- Detailed investigation of a particular parameter combination
- One-off evaluations without running a full phase

## Basic Usage

```bash
python run_single_evaluation.py cohere engineering 4 \
  templates/engineering_template_3.txt templates/marketing_template_2.txt \
  all-mpnet-base-v2 128 0 --ragas
```

This will:
1. Initialize a RAG system with the given configuration
2. Run an evaluation using LangSmith
3. Calculate detailed metrics
4. Save results to a structured output directory

## Command-Line Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `rag_type` | RAG system type: cohere or mistral |
| `team_type` | Team type: engineering or marketing |
| `top_k` | Number of documents to retrieve |
| `eng_template` | Path to engineering prompt template |
| `mkt_template` | Path to marketing prompt template |
| `embedding_model` | Name of the embedding model |
| `chunk_size` | Chunk size for text splitting |
| `chunk_overlap` | Chunk overlap for text splitting |

### Optional Parameters

| Parameter | Description |
|-----------|-------------|
| `--ragas` | Use RAGAS evaluations instead of standard metrics |
| `--retriever_type` | Retriever type: similarity (default), similarity_score_threshold, mmr, or multi_query |
| `--retriever_kwargs` | JSON string of retriever parameters (e.g. `'{"score_threshold": 0.8}'`) |
| `--limit` | Maximum number of questions to evaluate (default: 78) |
| `--output_dir` | Directory to save results (default: results/single_evaluations) |

## Examples

### Standard Evaluation

```bash
python run_single_evaluation.py cohere engineering 4 \
  templates/engineering_template_3.txt templates/marketing_template_2.txt \
  all-mpnet-base-v2 128 0
```

### RAGAS Evaluation

```bash
python run_single_evaluation.py cohere engineering 4 \
  templates/engineering_template_3.txt templates/marketing_template_2.txt \
  all-mpnet-base-v2 128 0 --ragas
```

### Using a Different Retriever Type

```bash
python run_single_evaluation.py cohere engineering 4 \
  templates/engineering_template_3.txt templates/marketing_template_2.txt \
  all-mpnet-base-v2 128 0 --ragas --retriever_type mmr \
  --retriever_kwargs '{"fetch_k": 10}'
```

### Limiting the Number of Questions

```bash
python run_single_evaluation.py cohere engineering 4 \
  templates/engineering_template_3.txt templates/marketing_template_2.txt \
  all-mpnet-base-v2 128 0 --ragas --limit 10
```

### Specifying an Output Directory

```bash
python run_single_evaluation.py cohere engineering 4 \
  templates/engineering_template_3.txt templates/marketing_template_2.txt \
  all-mpnet-base-v2 128 0 --ragas --output_dir ./my_custom_evaluation
```

## Output Format

Results are saved to a structured directory:
```
[output_dir]/[experiment_id]/results.json
```

The results.json file contains:
- Experiment configuration details
- Success status and execution time
- Detailed metrics:
  - Feedback metrics (accuracy, relevance, etc.)
  - Text comparison statistics (character length, word count)
  - TF-IDF similarity between generated and reference answers

## Evaluation Metrics

### Standard Metrics (default)

- correctness
- groundedness
- relevance
- retrieval_relevance

### RAGAS Metrics (with `--ragas` flag)

- ragas_answer_accuracy
- ragas_context_relevance
- ragas_faithfulness
- ragas_response_relevancy

## Memory Management

The script includes built-in memory management to help prevent CUDA out-of-memory errors:

- Cleans up RAGAS embedding models after use
- Forces garbage collection after evaluation
- Clears CUDA cache when applicable

## Comparing with Phased Evaluations

While the phased evaluation approach (`run_phased_evaluations.py`) helps systematically find optimal configurations, `run_single_evaluation.py` is useful when:

1. You want to test a very specific configuration that wasn't covered in the phased evaluation
2. You want to run a quick test with a small subset of questions
3. You're debugging a particular issue with a specific configuration
4. You want to compare a custom configuration against the best from the phased approach

## Analyzing Results

The results from `run_single_evaluation.py` can be analyzed using the same approach as for phased evaluations:

```bash
python analyze_top_performers.py /path/to/results.json
```

Or you can manually inspect the results.json file to see the detailed metrics.