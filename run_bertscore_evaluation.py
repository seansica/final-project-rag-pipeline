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
from bert_score import score
import torch
from loguru import logger
from dotenv import load_dotenv
import sys


def bert_score_evaluator(inputs: dict, outputs: dict) -> float:
    """Evaluates the similarity between the RAG answer and ground truth using BERTScore."""
    # Extract the necessary information
    question = inputs["question"]
    generated_answer = outputs["answer"]
    
    # Log the inputs structure
    logger.info(f"Inputs structure: {inputs.keys()}")
    
    try:
        # Clear CUDA cache before evaluation to help with memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Get reference answer directly from the dataset
        client = Client()
        
        # Use a fixed dataset name based on the experiment
        dataset_name = "w267-rag-validation-engineering"  # Default to engineering
        if "team_type" in inputs:
            team_type = inputs["team_type"]
            dataset_name = f"w267-rag-validation-{team_type}"
            
        logger.info(f"Searching for reference in dataset: {dataset_name}")
        
        # Get dataset examples
        dataset = client.read_dataset(dataset_name=dataset_name)
        examples = list(client.list_examples(dataset_id=dataset.id))
        
        # Find matching example for this question
        reference_answer = None
        for example in examples:
            example_data = example.dict()
            if example_data.get("inputs", {}).get("question") == question:
                reference_answer = example_data.get("outputs", {}).get("answer")
                logger.info(f"Found reference answer for question")
                break
        
        if not reference_answer:
            logger.warning(f"No reference answer found in dataset for question: {question[:50]}...")
            return 0.0
        
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

    load_dotenv()
    
    # Run the evaluation
    result = evaluate(
        args.experiment_name, 
        evaluators=[bert_score_evaluator]
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