# RAG System Experimentation Plan

This document summarizes the planned experiments for optimizing my RAG system across three phases.

## Phase 1: Embedding Model Comparison

Phase 1 tests different embedding models with fixed baseline parameters.

| # | LLM | Embedding Model | Chunk Size | Chunk Overlap | Top K | Retriever Type |
|---|-----|----------------|------------|---------------|-------|---------------|
| 1 | cohere | multi-qa-mpnet-base-dot-v1 | 128 | 0 | 4 | similarity |
| 2 | cohere | all-mpnet-base-v2 | 128 | 0 | 4 | similarity |
| 3 | cohere | all-MiniLM-L6-v2 | 128 | 0 | 4 | similarity |
| 4 | mistral | multi-qa-mpnet-base-dot-v1 | 128 | 0 | 4 | similarity |
| 5 | mistral | all-mpnet-base-v2 | 128 | 0 | 4 | similarity |
| 6 | mistral | all-MiniLM-L6-v2 | 128 | 0 | 4 | similarity |

NOTE TO SELF:
- All Engineering experiments are done
- Just need to run Mistral + Marketing experiments (3) to complete phase 1

## Phase 2: Chunk Size and Overlap Optimization

Phase 2 tests different chunk sizes and overlaps with the best embedding model from Phase 1.

| # | LLM | Embedding Model | Chunk Size | Chunk Overlap | Top K | Retriever Type |
|---|-----|----------------|------------|---------------|-------|---------------|
| 1 | cohere | {best_embedding_model} | 128 | 0 | 4 | similarity |
| 2 | cohere | {best_embedding_model} | 128 | 50 | 4 | similarity |
| 3 | cohere | {best_embedding_model} | 256 | 0 | 4 | similarity |
| 4 | cohere | {best_embedding_model} | 256 | 50 | 4 | similarity |
| 5 | cohere | {best_embedding_model} | 512 | 0 | 4 | similarity |
| 6 | cohere | {best_embedding_model} | 512 | 50 | 4 | similarity |
| 7 | mistral | {best_embedding_model} | 128 | 0 | 4 | similarity |
| 8 | mistral | {best_embedding_model} | 128 | 50 | 4 | similarity |
| 9 | mistral | {best_embedding_model} | 256 | 0 | 4 | similarity |
| 10 | mistral | {best_embedding_model} | 256 | 50 | 4 | similarity |
| 11 | mistral | {best_embedding_model} | 512 | 0 | 4 | similarity |
| 12 | mistral | {best_embedding_model} | 512 | 50 | 4 | similarity |

## Phase 3: Retriever Method Optimization

Phase 3 tests different retriever methods with the best embedding model and chunking strategy from previous phases.

| # | LLM | Retriever Type | Top K | Additional Parameters | Embedding Model | Chunk Size | Chunk Overlap |
|---|-----|---------------|-------|----------------------|----------------|------------|---------------|
| 1 | cohere | similarity | 5 | N/A | {best_embedding_model} | {best_chunk_size} | {best_chunk_overlap} |
| 2 | cohere | similarity | 10 | N/A | {best_embedding_model} | {best_chunk_size} | {best_chunk_overlap} |
| 3 | cohere | similarity_score_threshold | 4 | threshold=0.5 | {best_embedding_model} | {best_chunk_size} | {best_chunk_overlap} |
| 4 | cohere | similarity_score_threshold | 4 | threshold=0.8 | {best_embedding_model} | {best_chunk_size} | {best_chunk_overlap} |
| 5 | cohere | mmr | 4 | fetch_k=10 | {best_embedding_model} | {best_chunk_size} | {best_chunk_overlap} |
| 6 | cohere | multi_query | 4 | llm=cohere | {best_embedding_model} | {best_chunk_size} | {best_chunk_overlap} |
| 7 | mistral | similarity | 5 | N/A | {best_embedding_model} | {best_chunk_size} | {best_chunk_overlap} |
| 8 | mistral | similarity | 10 | N/A | {best_embedding_model} | {best_chunk_size} | {best_chunk_overlap} |
| 9 | mistral | similarity_score_threshold | 4 | threshold=0.5 | {best_embedding_model} | {best_chunk_size} | {best_chunk_overlap} |
| 10 | mistral | similarity_score_threshold | 4 | threshold=0.8 | {best_embedding_model} | {best_chunk_size} | {best_chunk_overlap} |
| 11 | mistral | mmr | 4 | fetch_k=10 | {best_embedding_model} | {best_chunk_size} | {best_chunk_overlap} |
| 12 | mistral | multi_query | 4 | llm=mistral | {best_embedding_model} | {best_chunk_size} | {best_chunk_overlap} |

## Workflow

1. Run Phase 1 and analyze results to determine the best embedding model:
   ```bash
   uv run python run_experiment_suite.py --phase 1 --max_parallel 2
   ```

2. Run Phase 2 with the best embedding model from Phase 1:
   ```bash
   uv run python run_experiment_suite.py --phase 2 --embedding_model "best_model" --max_parallel 2
   ```

3. Run Phase 3 with the best parameters from Phases 1 and 2:
   ```bash
   uv run python run_experiment_suite.py --phase 3 --embedding_model "best_model" --chunk_size XXX --chunk_overlap YYY --max_parallel 2
   ```

Note: This table omits the audience type (engineering vs. marketing) as both types are tested for every configuration.