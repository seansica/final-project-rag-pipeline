"""Evaluator for RAG system faithfulness using DeepEval's FaithfulnessMetric."""

from langsmith.schemas import Example, Run
from loguru import logger
import os
import sys

try:
    from deepeval import evaluate as deepeval_evaluate
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    logger.warning("DeepEval not installed. Install with 'pip install deepeval'")
    DEEPEVAL_AVAILABLE = False

def deepeval_faithfulness(run: Run, example: Example) -> float:
    """
    Evaluates the faithfulness of the RAG answer compared to the retrieved context
    using DeepEval's FaithfulnessMetric.
    
    Args:
        run: Run object from LangSmith
        example: Example object from LangSmith
        
    Returns:
        Float score between 0 and 1 (higher is better)
    """
    
    if not DEEPEVAL_AVAILABLE:
        logger.error("DeepEval not available. Cannot run faithfulness evaluation.")
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
        
        # Create DeepEval test case
        test_case = LLMTestCase(
            input=question,
            actual_output=generated_answer,
            retrieval_context=context  # For FaithfulnessMetric, use retrieval_context
        )
        
        # Set a threshold for faithfulness (higher is better)
        metric = FaithfulnessMetric(
            threshold=0.7,
            include_reason=True
        )
        
        # Measure faithfulness
        logger.info("Calculating Faithfulness score...")
        metric.measure(test_case)
        
        # Get the score
        faithfulness_score = metric.score
        
        logger.info(f"Faithfulness score: {faithfulness_score} (higher is better)")
        logger.info(f"Reason: {metric.reason}")
        
        return faithfulness_score
            
    except Exception as e:
        logger.error(f"Error in faithfulness evaluation: {str(e)}", exc_info=True)
        return 0.0