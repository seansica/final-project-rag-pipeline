# Final Project: RAG Pipeline

A comprehensive Retrieval Augmented Generation (RAG) evaluation system built with LangChain, LangSmith, and Qdrant. This project provides a systematic way to optimize RAG systems through a phased evaluation approach.

## Overview

This project implements:

1. **RAG System Pipeline**: Built with LangChain and Qdrant for vector storage
2. **Phased Evaluation Framework**: Three-phase optimization process
   - Phase 1: Embedding Model Selection
   - Phase 2: Chunking Strategy Selection
   - Phase 3: Retriever Method Selection
3. **Evaluation Metrics**: Support for both standard and RAGAS metrics
4. **Analysis Tools**: Scripts to analyze and visualize results

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/w267-final-project-rag-pipeline.git
cd w267-final-project-rag-pipeline

# Create a virtual environment
uv init --lib
source .venv/bin/activate
```

## Environment Setup

Create a `.env` file in the root directory with your API keys:

```
COHERE_API_KEY_PROD=your_cohere_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=your_langsmith_project
OPENAI_API_KEY=your_openai_api_key  # For evaluations
```

## Guides

Detailed guides for using the project components:

- [Phased RAG Evaluation Guide](guides/run_phased_evaluations_guide.md): How to run the phased evaluation pipeline
- [Single RAG Evaluation Guide](guides/run_single_evaluation_guide.md): How to evaluate a specific RAG configuration
- [Top Performer Analysis Guide](guides/analyze_top_performers_guide.md): How to analyze experiment results
- [Running Missing Experiments Guide](guides/run_single_missing_experiment_guide.md): How to recover from incomplete experiment runs

## Usage

### Running a Phased Evaluation

The phased evaluation systematically tests different RAG configurations:

```bash
# Phase 1: Embedding Model Selection
python run_phased_evaluations.py --phase 1 --max_parallel 2 --ragas

# Phase 2: Chunking Strategy Selection (after analyzing Phase 1)
python run_phased_evaluations.py --phase 2 --embedding_model "all-mpnet-base-v2" --max_parallel 2 --ragas

# Phase 3: Retriever Method Selection (after analyzing Phase 2)
python run_phased_evaluations.py --phase 3 --embedding_model "all-mpnet-base-v2" --chunk_size 512 --chunk_overlap 50 --max_parallel 2 --ragas
```

### Analyzing Results

After running experiments, analyze the results to find the best configurations:

```bash
# Analyze results from a specific phase
python analyze_top_performers.py results/phase1_ragas_*/results.json --output_dir ./analysis --visualize
```

### Running a Single Evaluation

To evaluate a specific RAG configuration:

```bash
python run_single_evaluation.py cohere engineering 4 \
  templates/engineering_template_3.txt templates/marketing_template_2.txt \
  all-mpnet-base-v2 128 0 --ragas
```

### Recovering Interrupted Experiments

To run a missing experiment from an incomplete experiment suite:

```bash
python run_single_missing_experiment.py --results_file /path/to/results.json
```

### Simple RAG Example

For a quick demo of the RAG system:

```bash
python examples/simple_rag.py
```

## Project Structure

- `src/rag267/`: Core RAG system implementation
  - `rag.py`: Main RAG pipeline implementation
  - `vectordb/`: Vector database components
  - `evals/`: Evaluation metrics and utilities
  - `data_sources.py`: Data source management
- `scripts/`: Utility scripts
- `templates/`: Prompt templates for different use cases
- `results/`: Experiment results (auto-generated)
- `examples/`: Example scripts demonstrating system usage
- `run_phased_evaluations.py`: Main script for running evaluations
- `analyze_top_performers.py`: Analysis script for experiment results
- `run_single_missing_experiment.py`: Script for running individual experiments
- `guides/`: Detailed guides for using the system

## Key Features

- **Multi-phase Optimization**: Systematically optimize RAG components
- **Multiple Embedding Models**: Support for various embedding models
- **Chunking Strategies**: Test various chunk sizes and overlaps
- **Retriever Methods**: Compare similarity, MMR, and multi-query approaches
- **Comprehensive Metrics**: Includes accuracy, faithfulness, relevance, and more
- **Visualization Tools**: Generate charts to interpret experiment results
- **Parallel Execution**: Run multiple experiments concurrently

## Dependencies

- langchain
- langsmith
- qdrant-client
- cohere
- sentence-transformers
- ragas
- pandas, numpy, matplotlib, seaborn
- and more (see pyproject.toml)