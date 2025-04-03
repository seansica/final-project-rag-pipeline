from typing import Dict, List, Any
from ragas.metrics import AnswerRelevancy
from ragas.llms import LangchainLLM
from langchain_openai import ChatOpenAI

def ragas_response_relevancy(inputs: dict, outputs: dict) -> float:
    """Evaluates the relevancy of the RAG answer to the original question using Ragas.
    
    Response Relevancy measures how well the generated answer addresses the original question's intent,
    by generating questions from the answer and computing similarity to the original question.
    
    Args:
        inputs: Dictionary containing input parameters with:
            - question: The original question asked
        outputs: Dictionary containing the RAG output, with keys:
            - answer: The generated response from the RAG system
            - documents: List of retrieved documents (not used for this metric)
            
    Returns:
        float: Response relevancy score between 0 and 1, where 1 means perfectly relevant
    """
    # Extract the question and answer
    question = inputs["question"]
    answer = outputs["answer"]
    
    # Initialize OpenAI model for evaluation
    llm = ChatOpenAI(model="gpt-4o")
    # Wrap the LLM with Ragas LLM Wrapper
    ragas_llm = LangchainLLM(llm)
    
    # Initialize the Answer Relevancy scorer
    answer_relevancy_scorer = AnswerRelevancy(llm=ragas_llm)
    
    # Create evaluation dataset with the expected format
    eval_dataset = [{
        "question": question,
        "answer": answer
    }]
    
    # Calculate answer relevancy score
    score = answer_relevancy_scorer.score(eval_dataset)
    
    # Extract the answer_relevancy score from the result
    return float(score["answer_relevancy"].iloc[0])