from ragas.metrics import ContextRelevance
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.dataset_schema import SingleTurnSample
import asyncio
from loguru import logger
import torch

def ragas_context_relevance(inputs: dict, outputs: dict) -> float:
    """Evaluates the relevance of retrieved contexts to the original query."""
    # Extract the necessary information
    question = inputs["question"]
    
    # Convert LangChain documents to text contexts that Ragas expects
    contexts = [doc.page_content for doc in outputs.get("documents", [])]
    
    if not contexts:
        logger.warning("No contexts retrieved for evaluation")
        return 0.0
    
    logger.debug(f'Number of contexts retrieved: {len(contexts)}');
    
    try:
        # Clear CUDA cache before evaluation to help with memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Initialize OpenAI model for evaluation 
        llm = ChatOpenAI(model="gpt-4o")
        # Wrap the LLM with Ragas LLM Wrapper
        ragas_llm = LangchainLLMWrapper(llm)
        
        # Initialize the Context Relevancy scorer
        scorer = ContextRelevance(llm=ragas_llm)
        
        # Create a sample with the expected format
        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts
        )
        
        # Run the async evaluation in a synchronous context
        logger.info(f"Evaluating context relevance for question: {question[:50]}...")
        score = asyncio.run(scorer.single_turn_ascore(sample))
        logger.info(f"Context relevance score: {score}")
        
        return float(score)
            
    except Exception as e:
        logger.error(f"Error in ragas_context_relevance: {str(e)}", exc_info=True)
        
        # Clear CUDA cache on error to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return 0.0