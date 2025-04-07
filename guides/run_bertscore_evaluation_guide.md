# BERTScore Evaluation Guide

This guide explains how to use the `run_bertscore_evaluation.py` script to perform post-hoc evaluation of RAG experiments using BERTScore.

## What is BERTScore?

BERTScore is a text evaluation metric that uses contextual embeddings from BERT to compare generated text against reference text. Unlike traditional metrics like BLEU or ROUGE which rely on exact word overlap, BERTScore captures semantic similarity using pre-trained language models, making it better suited for evaluating natural language generation tasks.

## Prerequisites

- Active LangSmith account with API access
- Previously run RAG experiments in LangSmith
- Reference answers available in a LangSmith dataset

## Setup

Make sure your environment variables are set up correctly:
   - `LANGSMITH_API_KEY`
   - `LANGSMITH_ENDPOINT` (typically https://api.smith.langchain.com)
   - `LANGSMITH_PROJECT`

## How it Works

The BERTScore evaluator works by:

1. Taking existing experiment outputs from LangSmith
2. Finding the corresponding reference answer in the validation dataset
3. Computing BERTScore between the generated answer and reference answer
4. Recording the evaluation results back in LangSmith

This approach lets you evaluate experiments after they've been run, without re-executing the RAG pipeline.

## Running the Evaluation

To evaluate an experiment, run:

```bash
uv run python run_bertscore_evaluation.py "experiment-name-or-id"
```

For example:
```bash
uv run python run_bertscore_evaluation.py "ragas-rag-cohere-engineering-emb-all-mpnet-base-v2-cs2048-co50-k4-mmr"
```

## Interpreting Results

The script will:

1. Log the evaluation process in both the console and `bert_score_eval.log`
2. Calculate BERTScore for each example in the experiment
3. Output summary statistics including:
   - Average BERTScore across all examples
   - Distribution of scores across different ranges (0-0.25, 0.25-0.5, etc.)

BERTScore values range from 0 to 1, where:
- 1.0 indicates perfect semantic match
- Values above 0.85 typically indicate strong semantic similarity
- Values below 0.5 suggest significant semantic differences

## Viewing Results in LangSmith

After running the evaluation:

1. Go to your LangSmith project
2. Find the original experiment
3. The BERTScore feedback should now be available alongside the original metrics

## Troubleshooting

If you encounter issues:

- Verify the experiment ID exists in your LangSmith project
- Ensure reference answers are available in the dataset
- Check GPU memory if using CUDA (the script attempts to manage memory but may need adjustments for large models)

## Extending the Evaluator

The BERTScore evaluator can be modified it to:

- Use different model variants (e.g., RoBERTa instead of BERT)
- Adjust preprocessing steps
- Combine with other metrics
- Change how reference answers are retrieved

## Notes on Reference Answers

The script automatically looks for reference answers in datasets named:
- `w267-rag-validation-engineering` for engineering questions
- `w267-rag-validation-marketing` for marketing questions

Make sure your validation data is properly organized in these datasets.