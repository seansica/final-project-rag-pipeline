#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langsmith",
#     "bert-score",
#     "torch",
#     "transformers",
#     "loguru",
# ]
# ///

from langsmith import Client, evaluate
from langsmith.beta._evals import compute_test_metrics
from langsmith.schemas import Example, Run
from bert_score import score
import torch
from loguru import logger
from dotenv import load_dotenv
import sys


load_dotenv()
client = Client()


def bert_score_evaluator(run: Run, example: Example) -> float:
    """Evaluates the similarity between the RAG answer and ground truth using BERTScore."""
    
    # Extract the necessary information
    question = run.inputs['inputs']['question']

    reference_answer = example.outputs["answer"]
    generated_answer = run.outputs["answer"]

    if not reference_answer:
            logger.warning(f"No reference answer found in dataset for question: {question[:50]}...")
            return 0.0
        
    try:
        # Clear CUDA cache before evaluation to help with memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log for debugging
        logger.info(f"Question: {question[:50]}...")
        logger.info(f"Generated answer: {generated_answer[:50]}...")
        logger.info(f"Reference answer: {reference_answer[:50]}...")
        
        # Calculate BERTScore
        logger.info("Calculating BERTScore...")
        with torch.no_grad():
            P, R, F1 = score([generated_answer], [reference_answer], lang="en", verbose=False)
        
        bert_score_value = float(F1[0])
        logger.info(f"BERTScore: {bert_score_value}")

        client.create_feedback(
            run_id=run.id,
            key="BERTScore",
            score=bert_score_value,
        )
        
        return bert_score_value
            
    except Exception as e:
        logger.error(f"Error in bert_score_evaluator: {str(e)}", exc_info=True)
        
        # Clear CUDA cache on error to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return 0.0

def main():
    import argparse
    
    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("bert_score_eval.log", level="DEBUG", rotation="10 MB")
    
    parser = argparse.ArgumentParser(description="Evaluate an experiment with BERTScore")
    parser.add_argument("experiment_name", help="LangSmith experiment name or ID")
    args = parser.parse_args()
    
    logger.info(f"Evaluating experiment: {args.experiment_name}")

    # Run the evaluation
    result = evaluate(
        args.experiment_name, 
        evaluators=[bert_score_evaluator],
        client=client
    )
    
    # Print evaluation results
    logger.info("\nEvaluation Results:")
    df = result.to_pandas()
    logger.info(f"Number of runs evaluated: {len(df)}")
    
    # Find the BERTScore column
    bert_score_cols = [col for col in df.columns if 'bert_score' in col.lower()]
    if bert_score_cols:
        scores = df[bert_score_cols[0]].dropna().tolist()
        if scores:
            avg_score = sum(scores) / len(scores)
            logger.info(f"Average BERTScore: {avg_score:.4f}")
            
            # Distribution of scores
            bins = [0.0, 0.25, 0.5, 0.75, 0.85, 0.95, 1.0]
            hist = [0] * (len(bins)-1)
            for s in scores:
                for i in range(len(bins)-1):
                    if bins[i] <= s < bins[i+1]:
                        hist[i] += 1
                        break
            
            logger.info("\nDistribution of BERTScores:")
            for i in range(len(bins)-1):
                logger.info(f"{bins[i]:.2f}-{bins[i+1]:.2f}: {hist[i]} examples")
    else:
        logger.info("No BERTScore results found")

if __name__ == "__main__":
    main()