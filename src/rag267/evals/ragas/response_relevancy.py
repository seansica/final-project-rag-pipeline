from ragas.metrics import ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.dataset_schema import SingleTurnSample
from langchain_community.embeddings import HuggingFaceEmbeddings
import asyncio
from loguru import logger


def ragas_response_relevancy(inputs: dict, outputs: dict) -> float:
    """Evaluates the relevancy of the RAG answer to the original question."""
    # Extract the question and answer
    question = inputs["question"]
    answer = outputs.get("answer", "")
    
    if not answer:
        logger.warning("No answer provided for response relevancy evaluation")
        return 0.0
    
    try:
        # Initialize OpenAI model for evaluation
        llm = ChatOpenAI(model="gpt-4o")
        # Wrap the LLM with Ragas LLM Wrapper
        ragas_llm = LangchainLLMWrapper(llm)
        
        # Initialize embeddings directly instead of using global variable
        logger.info("Initializing embeddings model")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # Initialize the Response Relevancy scorer
        scorer = ResponseRelevancy(llm=ragas_llm, embeddings=embeddings)
        
        # Create a sample with the expected format
        sample = SingleTurnSample(
            user_input=question,
            response=answer
        )
        
        # Run the async evaluation in a synchronous context
        logger.info(f"Evaluating response relevancy for question: {question[:50]}...")
        score = asyncio.run(scorer.single_turn_ascore(sample))
        logger.info(f"Response relevancy score: {score}")
        
        return float(score)
            
    except Exception as e:
        logger.error(f"Error in ragas_response_relevancy: {str(e)}", exc_info=True)
        return 0.0