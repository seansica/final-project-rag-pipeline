"""BERTScore evaluator for RAG system evaluation."""

from langsmith.schemas import Example, Run
from loguru import logger
import torch

try:
    from bert_score import score
    BERTSCORE_AVAILABLE = True
except ImportError:
    logger.warning("BERTScore not installed. Install with 'pip install bert-score'")
    BERTSCORE_AVAILABLE = False

def bertscore_evaluator(run: Run, example: Example) -> float:
    """
    Evaluates the similarity between the RAG answer and ground truth using BERTScore.
    
    Args:
        run: Run object from LangSmith
        example: Example object from LangSmith
        
    Returns:
        Float score between 0 and 1 (higher is better)
    """
    
    if not BERTSCORE_AVAILABLE:
        logger.error("BERTScore not available. Cannot run evaluation.")
        return 0.0
    
    # Extract the question from the inputs
    question = ""
    if isinstance(run.inputs, dict) and 'inputs' in run.inputs:
        if isinstance(run.inputs['inputs'], dict) and 'question' in run.inputs['inputs']:
            question = run.inputs['inputs']['question']
    
    if not question:
        logger.warning("No question found in run inputs")
        return 0.0

    # Get the expected answer from the reference
    reference_answer = example.outputs.get('answer', '')
    if not reference_answer:
        logger.warning(f"No reference answer found in dataset for question: {question[:50]}...")
        return 0.0
    
    # Get the generated answer from outputs
    generated_answer = ""
    if isinstance(run.outputs, dict) and 'answer' in run.outputs:
        generated_answer = run.outputs['answer']
    
    if not generated_answer:
        logger.warning(f"No generated answer found for question: {question[:50]}...")
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
        
        return bert_score_value
            
    except Exception as e:
        logger.error(f"Error in bertscore_evaluator: {str(e)}", exc_info=True)
        
        # Clear CUDA cache on error to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return 0.0