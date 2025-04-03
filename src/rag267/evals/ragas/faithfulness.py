from typing import Dict, List, Any
from langchain_core.documents import Document
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLM
from langchain_openai import ChatOpenAI

def ragas_faithfulness(inputs: dict, outputs: dict) -> float:
    """Evaluates the faithfulness of the RAG answer using Ragas.
    
    Faithfulness measures if the generated answer contains only information
    that can be found in the retrieved context documents.
    
    Args:
        inputs: Dictionary containing input parameters with:
            - question: The original question asked
        outputs: Dictionary containing the RAG output, with keys:
            - answer: The generated response from the RAG system
            - documents: List of retrieved documents used for generating the answer
            
    Returns:
        float: Faithfulness score between 0 and 1, where 1 means perfectly faithful
    """
    # Extract the answer and documents from the outputs
    answer = outputs["answer"]
    question = inputs["question"]
    
    # Convert LangChain documents to the format expected by Ragas
    context_docs = [doc.page_content for doc in outputs["documents"]]
    
    # Initialize OpenAI model for evaluation
    llm = ChatOpenAI(model="gpt-4o")
    # Wrap the LLM with Ragas LLM Wrapper
    ragas_llm = LangchainLLM(llm)
    
    # Initialize the Faithfulness scorer
    faithfulness_scorer = Faithfulness(llm=ragas_llm)
    
    # Create evaluation dataset with the expected format
    eval_dataset = [{
        "question": question,
        "contexts": context_docs,
        "answer": answer
    }]
    
    # Calculate faithfulness score
    score = faithfulness_scorer.score(eval_dataset)
    
    # Extract the faithfulness score from the result
    return float(score["faithfulness"].iloc[0])