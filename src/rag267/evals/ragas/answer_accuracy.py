from ragas.metrics import AnswerAccuracy
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.dataset_schema import SingleTurnSample
import asyncio
import logging
from langsmith import Client
from loguru import logger


def ragas_answer_accuracy(inputs: dict, outputs: dict) -> float:
    """Evaluates the accuracy of the RAG answer against ground truth using Ragas."""
    # Extract the necessary information
    question = inputs["question"]
    generated_answer = outputs["answer"]
    
    # Log the inputs structure
    logger.info(f"Inputs structure: {inputs.keys()}")
    
    try:
        # Approach: Get reference answer directly from the dataset
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
        
        # Initialize OpenAI model for evaluation 
        llm = ChatOpenAI(model="gpt-4o")
        # Wrap the LLM with Ragas LLM Wrapper
        ragas_llm = LangchainLLMWrapper(llm)
        
        # Initialize the Answer Accuracy scorer
        scorer = AnswerAccuracy(llm=ragas_llm)
        
        # Create a sample with the expected format
        sample = SingleTurnSample(
            user_input=question,
            response=generated_answer,
            reference=reference_answer
        )
        
        # Run the async evaluation in a synchronous context
        logger.info(f"Evaluating answer accuracy...")
        score = asyncio.run(scorer.single_turn_ascore(sample))
        logger.info(f"Answer accuracy score: {score}")
        
        return float(score)
            
    except Exception as e:
        logger.error(f"Error in ragas_answer_accuracy: {str(e)}", exc_info=True)
        return 0.0