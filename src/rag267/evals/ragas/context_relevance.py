from typing import Dict, List, Any
from ragas.metrics import ContextRelevancy
from ragas.llms import LangchainLLM
from langchain_openai import ChatOpenAI

def ragas_context_relevance(inputs: dict, outputs: dict) -> float:
    """Evaluates the relevance of retrieved contexts to the original query using Ragas.
    
    Context Relevance measures how well the retrieved contexts relate to the user's query.
    It uses two LLM-as-judge prompts to rate context relevance and converts ratings to a [0,1] scale.
    
    Rating scale:
    - 0: Retrieved contexts are not relevant at all
    - 1: Contexts are partially relevant
    - 2: Contexts are completely relevant
    
    Args:
        inputs: Dictionary containing input parameters with:
            - question: The original question asked
        outputs: Dictionary containing the RAG output with:
            - documents: List of retrieved documents used for generating the answer
            
    Returns:
        float: Context relevance score between 0 and 1, where 1 means perfectly relevant contexts
    """
    # Extract the necessary information
    question = inputs["question"]
    
    # Convert LangChain documents to text contexts that Ragas expects
    contexts = [doc.page_content for doc in outputs["documents"]]
    
    # If no contexts were retrieved, return 0 for relevance
    if not contexts:
        return 0.0
    
    # Initialize OpenAI model for evaluation 
    llm = ChatOpenAI(model="gpt-4o")
    # Wrap the LLM with Ragas LLM Wrapper
    ragas_llm = LangchainLLM(llm)
    
    # Initialize the Context Relevancy scorer
    context_relevancy_scorer = ContextRelevancy(llm=ragas_llm)
    
    # Create a dataset with the expected format
    eval_dataset = [{
        "question": question,
        "contexts": contexts
    }]
    
    # Calculate context relevance score
    score = context_relevancy_scorer.score(eval_dataset)
    
    # Extract the context_relevance score from the result
    return float(score["context_relevancy"].iloc[0])