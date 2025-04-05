# Top Performer Analysis Guide

This guide explains how to use the `analyze_top_performers.py` script to analyze experiment results and identify the top performing RAG configurations.

## Overview

The `analyze_top_performers.py` script analyzes a `results.json` file from your RAG experiments and determines:

1. The top performing system for each metric (accuracy, faithfulness, etc.)
2. The systems with output closest to the reference length
3. The overall best system based on average ranking across all metrics

## Basic Usage

```bash
python analyze_top_performers.py /path/to/results.json
```

This will:
- Load and analyze the results file
- Calculate the top performer for each metric
- Compute an overall score based on normalized rankings
- Print a detailed report to the console

## Command-Line Parameters

| Parameter | Description |
|-----------|-------------|
| `results_file` | Required. Path to the results.json file |
| `--output_dir` | Directory to save analysis outputs (optional) |
| `--visualize` | Create visualizations of the results (optional) |

## Output Analysis

### Console Output

The script outputs a detailed analysis to the console with sections for:

1. **RAGAS Metrics** - Shows the top performer for each metric like answer accuracy and faithfulness
2. **Text Comparison Metrics** - Shows which systems produce output closest to reference length
3. **Overall Top Performer** - The system with the best average ranking across all metrics

Example output:
```
TOP PERFORMING RAG SYSTEMS BY METRIC
================================================================================

RAGAS METRICS (higher is better):
--------------------------------------------------

RAGAS_ANSWER_ACCURACY:
  Score: 0.3173
  Experiment: ragas-rag-cohere-engineering-emb-all-mpnet-base-v2-cs128-co0-k4
  Configuration:
    - RAG Type: cohere
    - Team Type: engineering
    - Embedding Model: all-mpnet-base-v2
    - Chunk Size: 128
    - Chunk Overlap: 0
    - Top K: 4
    - Retriever Type: similarity

...

OVERALL TOP PERFORMER (based on average ranking across all metrics):
--------------------------------------------------
  Overall Score: 0.9235
  Experiment: ragas-rag-cohere-engineering-emb-all-mpnet-base-v2-cs128-co0-k4
  ...
```

### Saved Analysis Files

When using the `--output_dir` flag, these files are generated:

- `analysis.json`: Complete analysis data in JSON format
- `all_metrics.csv`: CSV file with all metrics for all experiments
- Visualization files (if `--visualize` is used)

## Visualizations

When the `--visualize` flag is used, the script generates these visualizations:

1. **Feedback Metrics Heatmap**: Shows all metrics across all experiments
2. **Overall Ranking Bar Chart**: Ranks experiments by overall score
3. **Top Performer Radar Chart**: Shows the performance profile of the best system

Example command to generate visualizations:
```bash
python analyze_top_performers.py results/phase1_ragas_*/results.json --output_dir ./analysis --visualize
```

## Metrics Interpretation

### RAGAS Metrics

For all RAGAS metrics, higher values are better:

- **ragas_answer_accuracy**: How accurate the generated answer is compared to the reference
- **ragas_context_relevance**: How relevant the retrieved context is to the question
- **ragas_faithfulness**: How faithful the answer is to the retrieved context
- **ragas_response_relevancy**: How relevant the answer is to the original question

### Text Comparison Metrics

For text comparison metrics, values closer to 1.0 indicate better length matching:

- **character_length**: The ratio of output character length to reference character length
- **word_count**: The ratio of output word count to reference word count

The "closeness score" measures how close the ratio is to 1.0 (higher is better).

## Ranking Methodology

The script calculates an overall ranking using these steps:

1. For each metric, calculate a normalized rank (0-1 scale, 1 is best)
2. Average the normalized ranks across all metrics to get an overall score
3. The experiment with the highest overall score is the top performer

This ensures fair comparison across metrics with different scales.

## Using Analysis Results

After identifying the best configurations:

1. **For Phase 1**: Use the best embedding model as input to Phase 2
   ```bash
   python run_phased_evaluations.py --phase 2 --embedding_model "best_model_from_analysis" --ragas
   ```

2. **For Phase 2**: Use the best chunking strategy as input to Phase 3
   ```bash
   python run_phased_evaluations.py --phase 3 --embedding_model "model" --chunk_size 256 --chunk_overlap 50 --ragas
   ```

3. **For Phase 3**: Implement the final optimized RAG configuration in your application

## Tips for Analysis

- **Compare across phases**: Analyze results between phases to validate improvements
- **Look for patterns**: Check if certain configurations perform consistently well
- **Consider trade-offs**: Some configurations may excel in some metrics but not others
- **Team type matters**: Engineering and marketing configurations often have different optimal settings