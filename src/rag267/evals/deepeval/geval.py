"""Evaluator for RAG system quality using DeepEval's GEval metric."""

from langsmith.schemas import Example, Run
from loguru import logger
import os
import sys

try:
    from deepeval import evaluate as deepeval_evaluate
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    DEEPEVAL_AVAILABLE = True
except ImportError:
    logger.warning("DeepEval not installed. Install with 'pip install deepeval'")
    DEEPEVAL_AVAILABLE = False

def deepeval_geval(run: Run, example: Example) -> float:
    """
    Evaluates the overall quality of the RAG answer using DeepEval's GEval metric,
    which uses LLM-as-a-judge with chain-of-thoughts.
    
    Args:
        run: Run object from LangSmith
        example: Example object from LangSmith
        
    Returns:
        Float score between 0 and 1 (higher is better)
    """
    
    if not DEEPEVAL_AVAILABLE:
        logger.error("DeepEval not available. Cannot run GEval evaluation.")
        return 0.0
    
    # Check if OpenAI API key is available (required for DeepEval)
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required for DeepEval to work")
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
        # Handle different document types
        if isinstance(doc, dict):
            # Dictionary-style documents
            if 'page_content' in doc:
                context.append(doc['page_content'])
            elif 'pageContent' in doc:
                context.append(doc['pageContent'])
            elif 'text' in doc:
                context.append(doc['text'])
            # Look one level deeper in case the document has a nested structure
            elif 'metadata' in doc and isinstance(doc['metadata'], dict):
                # Add metadata if it might contain useful information
                for key, value in doc['metadata'].items():
                    if isinstance(value, str) and len(value) > 50:  # Likely content
                        context.append(value)
        # Handle object-style documents with attributes
        elif hasattr(doc, 'page_content'):
            context.append(doc.page_content)
        elif hasattr(doc, 'pageContent'):
            context.append(doc.pageContent)
        elif hasattr(doc, 'text'):
            context.append(doc.text)
    
    if not context:
        # Print the structure of the retrieved documents to debug
        logger.warning(f"Could not extract context from retrieved documents for question: {question[:50]}...")
        logger.debug(f"Document structure: {retrieved_docs[:1]}")
        
        # Last resort: try to convert to string and use that
        try:
            for doc in retrieved_docs:
                doc_str = str(doc)
                # Only add if the string is substantial (not just object representation)
                if len(doc_str) > 100 and '<' not in doc_str[:10]:  # Avoid object repr strings
                    context.append(doc_str)
        except Exception as e:
            logger.error(f"Failed to extract context even as string: {e}")
            
        if not context:
            return 0.0
    
    try:
        # Log for debugging
        logger.info(f"Question: {question[:50]}...")
        logger.info(f"Generated answer: {generated_answer[:50]}...")
        logger.info(f"Context: {context[0][:50]}... (and {len(context)-1} more documents)")
        if expected_answer:
            logger.info(f"Expected answer: {expected_answer[:50]}...")
        
        # Create DeepEval test case
        test_case = LLMTestCase(
            input=question,
            actual_output=generated_answer,
            context=context
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
            geval_metric = GEval(
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
            geval_metric = GEval(
                name="RAG_Relevance",
                evaluation_steps=[
                    "Evaluate if the answer directly addresses the question asked",
                    "Check if the answer contains factual information from the provided context",
                    "Verify the answer doesn't include facts that contradict the provided context",
                    "The answer should be factual, grounded in the context, and not include hallucinated information"
                ],
                evaluation_params=evaluation_params,
                threshold=0.7,
                model="gpt-4",  # Using GPT-4 for more accurate evaluation
                include_reason=True
            )
        
        # Measure quality using GEval
        logger.info("Calculating GEval score...")
        geval_metric.measure(test_case)
        
        # Get the score
        geval_score = geval_metric.score
        
        logger.info(f"GEval score: {geval_score} (higher is better)")
        logger.info(f"Reason: {geval_metric.reason}")
        
        return geval_score
            
    except Exception as e:
        logger.error(f"Error in geval evaluation: {str(e)}", exc_info=True)
        return 0.0