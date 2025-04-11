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
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    DEEPEVAL_AVAILABLE = True
except ImportError:
    logger.warning("DeepEval not installed. Install with 'pip install deepeval'")
    DEEPEVAL_AVAILABLE = False

load_dotenv()
client = Client()

def geval_rag_relevance(run: Run, example: Example) -> float:
    """
    Evaluates the relevance and factual accuracy of RAG system responses 
    using DeepEval's GEval metric.
    """
    
    if not DEEPEVAL_AVAILABLE:
        logger.error("DeepEval not available. Cannot run GEval evaluation.")
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
    
    # Get the expected answer from the reference
    expected_answer = example.outputs.get('answer', '')
    if not expected_answer:
        logger.warning(f"No expected answer found in example for question: {question[:50]}...")
        # We can still evaluate without an expected answer, but it's less reliable
    
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
        if expected_answer:
            logger.info(f"Expected answer: {expected_answer[:50]}...")
        
        # DeepEval expects context to be a list of strings, not a single string
        # Create DeepEval test case
        test_case = LLMTestCase(
            input=question,
            actual_output=generated_answer,
            context=context  # Pass the list of context strings directly
        )
        
        # If we have an expected answer, add it to the test case
        if expected_answer:
            test_case.expected_output = expected_answer
        
        # Define evaluation criteria based on the available information
        evaluation_params = [
            LLMTestCaseParams.INPUT,  # question
            LLMTestCaseParams.ACTUAL_OUTPUT,  # generated answer
            LLMTestCaseParams.CONTEXT  # retrieved documents
        ]
        
        # If we have an expected answer, include it in the evaluation
        if expected_answer:
            evaluation_params.append(LLMTestCaseParams.EXPECTED_OUTPUT)
            
            # Define a custom metric that evaluates both relevance to the context and comparison to expected answer
            relevance_metric = GEval(
                name="RAG_Quality",
                evaluation_steps=[
                    "Evaluate if the answer directly addresses the question asked",
                    "Check if the answer contains factual information from the provided context",
                    "Verify the answer doesn't include facts that contradict the provided context",
                    "Compare the answer's quality to the expected answer, considering completeness and accuracy",
                    "The answer should be factual, grounded in the context, and not include hallucinated information"
                ],
                evaluation_params=evaluation_params,
                threshold=0.7,
                verbose_mode=True
            )
        else:
            # If no expected answer, just evaluate based on question and context
            relevance_metric = GEval(
                name="RAG_Relevance",
                evaluation_steps=[
                    "Evaluate if the answer directly addresses the question asked",
                    "Check if the answer contains factual information from the provided context",
                    "Verify the answer doesn't include facts that contradict the provided context",
                    "The answer should be factual, grounded in the context, and not include hallucinated information"
                ],
                evaluation_params=evaluation_params,
                threshold=0.7,
                verbose_mode=True
            )
        
        # Measure relevance using GEval
        logger.info("Calculating GEval score...")
        relevance_metric.measure(test_case)
        
        # Get the score
        geval_score = relevance_metric.score
        
        logger.info(f"GEval score: {geval_score} (higher is better)")
        logger.info(f"Reason: {relevance_metric.reason}")

        # Add feedback to LangSmith
        client.create_feedback(
            run_id=run.id,
            key="RAG_Quality",
            score=geval_score,
            comment=relevance_metric.reason
        )
        
        return geval_score
            
    except Exception as e:
        logger.error(f"Error in geval_rag_relevance: {str(e)}", exc_info=True)
        return 0.0

def main():
    import argparse
    
    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("geval_evaluation.log", level="DEBUG", rotation="10 MB")
    
    parser = argparse.ArgumentParser(description="Evaluate RAG experiment using DeepEval's GEval metric")
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
        evaluators=[geval_rag_relevance],
        client=client
    )
    
    # Print evaluation results
    logger.info("\nEvaluation Results:")
    df = result.to_pandas()
    logger.info(f"Number of runs evaluated: {len(df)}")
    
    # Find the GEval score column
    geval_cols = [col for col in df.columns if 'rag_quality' in col.lower() or 'rag_relevance' in col.lower()]
    if geval_cols:
        scores = df[geval_cols[0]].dropna().tolist()
        if scores:
            avg_score = sum(scores) / len(scores)
            logger.info(f"Average GEval score: {avg_score:.4f} (higher is better)")
            
            # Distribution of scores
            bins = [0.0, 0.25, 0.5, 0.75, 0.85, 0.95, 1.0]
            hist = [0] * (len(bins)-1)
            for s in scores:
                for i in range(len(bins)-1):
                    if bins[i] <= s < bins[i+1]:
                        hist[i] += 1
                        break
            
            logger.info("\nDistribution of GEval scores:")
            for i in range(len(bins)-1):
                logger.info(f"{bins[i]:.2f}-{bins[i+1]:.2f}: {hist[i]} examples")
    else:
        logger.info("No GEval results found")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())