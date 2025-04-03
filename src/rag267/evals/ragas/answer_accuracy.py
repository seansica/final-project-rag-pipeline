from typing import Dict, List, Any
from ragas.metrics import AnswerAccuracy
from ragas.llms import LangchainLLM
from langchain_openai import ChatOpenAI


def ragas_answer_accuracy(inputs: dict, outputs: dict) -> float:
    """Evaluates the accuracy of the RAG answer against ground truth using Ragas.
    
    Answer Accuracy measures how well a model's response aligns with a reference ground truth.
    It uses two LLM-as-judge prompts to rate response accuracy and converts ratings to a [0,1] scale.
    
    Rating scale:
    - 0: Response is inaccurate or doesn't address the same question
    - 2: Response partially aligns with the reference
    - 4: Response exactly aligns with the reference
    
    Args:
        inputs: Dictionary containing input parameters with:
            - question: The original question asked
            
        outputs: Dictionary containing the RAG output with:
            - answer: The generated response from the RAG system
            
    Returns:
        float: Answer accuracy score between 0 and 1, where 1 means perfect accuracy
    """
    # Extract the necessary information
    question = inputs["question"]
    generated_answer = outputs["answer"]
    
    # Get the reference answer from the inputs (LangSmith structure)
    # In the LangSmith format, the ground truth is in inputs["outputs"]["answer"]
    reference_answer = inputs["outputs"]["answer"] if "outputs" in inputs else None
    
    # If there's no reference answer, we can't evaluate accuracy
    if reference_answer is None:
        return 0.0
    
    # Initialize OpenAI model for evaluation 
    llm = ChatOpenAI(model="gpt-4o")
    # Wrap the LLM with Ragas LLM Wrapper
    ragas_llm = LangchainLLM(llm)
    
    # Initialize the Answer Accuracy scorer
    answer_accuracy_scorer = AnswerAccuracy(llm=ragas_llm)
    
    # Use the batch scoring method (which is synchronous unlike the single_turn_ascore method)
    # Create a dataset with the expected format
    eval_dataset = [{
        "question": question,
        "answer": generated_answer,
        "reference": reference_answer
    }]
    
    # Calculate answer accuracy score
    score = answer_accuracy_scorer.score(eval_dataset)
    
    # Extract the answer_accuracy score from the result
    return float(score["answer_accuracy"].iloc[0])