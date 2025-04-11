#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langsmith",
#     "deepeval",
#     "loguru",
#     "python-dotenv",
# ]
# ///

from langsmith import Client, evaluate
from langsmith.schemas import Example, Run
from loguru import logger
from dotenv import load_dotenv
import sys
import os

try:
    from deepeval import evaluate as deepeval_evaluate
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    logger.warning("DeepEval not installed. Install with 'pip install deepeval'")
    DEEPEVAL_AVAILABLE = False

load_dotenv()
client = Client()

def faithfulness_evaluator(run: Run, example: Example) -> float:
    """Evaluates the faithfulness of the RAG answer compared to the retrieved context."""
    
    if not DEEPEVAL_AVAILABLE:
        logger.error("DeepEval not available. Cannot run faithfulness evaluation.")
        return 0.0
    
    # Extract the question from the inputs
    question = ""
    if isinstance(run.inputs, dict) and 'inputs' in run.inputs:
        if isinstance(run.inputs['inputs'], dict) and 'question' in run.inputs['inputs']:
            question = run.inputs['inputs']['question']
    
    if not question:
        logger.warning("No question found in run inputs")
        return 0.0

    # Get the generated answer from outputs
    generated_answer = ""
    if isinstance(run.outputs, dict) and 'answer' in run.outputs:
        generated_answer = run.outputs['answer']
    
    if not generated_answer:
        logger.warning(f"No generated answer found for question: {question[:50]}...")
        return 0.0
    
    # Get the retrieved documents from outputs
    retrieved_docs = []
    if isinstance(run.outputs, dict) and 'documents' in run.outputs:
        retrieved_docs = run.outputs['documents']
    
    if not retrieved_docs:
        logger.warning(f"No retrieved documents found for question: {question[:50]}...")
        return 0.0
    
    # Extract the page content from the documents
    context = []
    for doc in retrieved_docs:
        if isinstance(doc, dict) and 'page_content' in doc:
            context.append(doc['page_content'])
    
    if not context:
        logger.warning(f"Could not extract context from retrieved documents for question: {question[:50]}...")
        return 0.0
    
    try:
        # Log for debugging
        logger.info(f"Question: {question[:50]}...")
        logger.info(f"Generated answer: {generated_answer[:50]}...")
        logger.info(f"Context: {context[0][:50]}... (and {len(context)-1} more documents)")
        
        # Create DeepEval test case - using retrieval_context for FaithfulnessMetric
        test_case = LLMTestCase(
            input=question,
            actual_output=generated_answer,
            retrieval_context=context  # For FaithfulnessMetric, use retrieval_context
        )
        
        # Set a threshold for faithfulness detection (higher is better)
        metric = FaithfulnessMetric(
            threshold=0.7,
            include_reason=True
        )
        
        # Measure faithfulness
        logger.info("Calculating Faithfulness score...")
        metric.measure(test_case)
        
        # DeepEval faithfulness score is already 0-1 where higher is better
        faithfulness_score = metric.score
        
        logger.info(f"Faithfulness score: {faithfulness_score} (higher is better)")
        logger.info(f"Reason: {metric.reason}")

        # Add feedback to LangSmith
        client.create_feedback(
            run_id=run.id,
            key="Faithfulness",
            score=faithfulness_score,
            comment=metric.reason
        )
        
        return faithfulness_score
            
    except Exception as e:
        logger.error(f"Error in faithfulness_evaluator: {str(e)}", exc_info=True)
        return 0.0

def main():
    import argparse
    
    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("faithfulness_eval.log", level="DEBUG", rotation="10 MB")
    
    parser = argparse.ArgumentParser(description="Evaluate RAG experiment for faithfulness")
    parser.add_argument("experiment_name", help="LangSmith experiment name or ID")
    args = parser.parse_args()
    
    if not DEEPEVAL_AVAILABLE:
        logger.error("DeepEval package is required. Install with 'pip install deepeval'")
        return 1
    
    # Check that OpenAI API key is available (required for DeepEval)
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required for DeepEval to work")
        return 1
    
    logger.info(f"Evaluating experiment: {args.experiment_name}")

    # Run the evaluation
    result = evaluate(
        args.experiment_name, 
        evaluators=[faithfulness_evaluator],
        client=client
    )
    
    # Print evaluation results
    logger.info("\nEvaluation Results:")
    df = result.to_pandas()
    logger.info(f"Number of runs evaluated: {len(df)}")
    
    # Find the faithfulness score column
    faithfulness_cols = [col for col in df.columns if 'faithfulness' in col.lower()]
    if faithfulness_cols:
        scores = df[faithfulness_cols[0]].dropna().tolist()
        if scores:
            avg_score = sum(scores) / len(scores)
            logger.info(f"Average Faithfulness score: {avg_score:.4f} (higher is better)")
            
            # Distribution of scores
            bins = [0.0, 0.25, 0.5, 0.75, 0.85, 0.95, 1.0]
            hist = [0] * (len(bins)-1)
            for s in scores:
                for i in range(len(bins)-1):
                    if bins[i] <= s < bins[i+1]:
                        hist[i] += 1
                        break
            
            logger.info("\nDistribution of Faithfulness scores:")
            for i in range(len(bins)-1):
                logger.info(f"{bins[i]:.2f}-{bins[i+1]:.2f}: {hist[i]} examples")
    else:
        logger.info("No faithfulness evaluation results found")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())