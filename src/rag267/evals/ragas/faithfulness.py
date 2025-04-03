from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.dataset_schema import SingleTurnSample
import asyncio
from loguru import logger

def ragas_faithfulness(inputs: dict, outputs: dict) -> float:
    """Evaluates the faithfulness of the RAG answer using Ragas."""
    # Extract the answer and documents from the outputs
    question = inputs["question"]
    answer = outputs.get("answer", "")
    
    # Convert LangChain documents to the format expected by Ragas
    contexts = [doc.page_content for doc in outputs.get("documents", [])]
    
    # If no answer or contexts, return 0
    if not answer or not contexts:
        logger.warning("Missing answer or contexts for faithfulness evaluation")
        return 0.0
    
    try:
        # Initialize OpenAI model for evaluation
        llm = ChatOpenAI(model="gpt-4o")
        # Wrap the LLM with Ragas LLM Wrapper
        ragas_llm = LangchainLLMWrapper(llm)
        
        # Initialize the Faithfulness scorer
        scorer = Faithfulness(llm=ragas_llm)
        
        # Create a sample with the expected format
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )
        
        # Run the async evaluation in a synchronous context
        logger.info(f"Evaluating faithfulness for question: {question[:50]}...")
        score = asyncio.run(scorer.single_turn_ascore(sample))
        logger.info(f"Faithfulness score: {score}")
        
        return float(score)
            
    except Exception as e:
        logger.error(f"Error in ragas_faithfulness: {str(e)}", exc_info=True)
        return 0.0