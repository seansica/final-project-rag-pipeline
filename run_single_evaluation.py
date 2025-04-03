#!/usr/bin/env python
"""
Wrapper script to run a single RAG evaluation with specific parameters.
Usage: 
  Standard evaluation:
  python run_single_evaluation.py cohere engineering 4 templates/engineering_template.txt templates/marketing_template.txt multi-qa-mpnet-base-dot-v1 128 0
  
  Ragas evaluation:
  python run_single_evaluation.py cohere engineering 4 templates/engineering_template.txt templates/marketing_template.txt multi-qa-mpnet-base-dot-v1 128 0 --ragas
"""
import sys
import json
import os
import time
import argparse
from dotenv import load_dotenv
from langsmith import Client
from langchain_openai import ChatOpenAI
from typing_extensions import Annotated, TypedDict
import logging

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import custom modules (using absolute imports)
from rag267.rag import RAGSystem
from rag267.vectordb.utils import Team, SupportedGeneratorModels
from rag267.vectordb.manager import VectorDatabaseManager
from rag267.data_sources import data_sources
from rag267.evals.correctness import correctness
from rag267.evals.relevance import relevance
from rag267.evals.retrieval_relevance import retrieval_relevance
from rag267.evals.groundedness import groundedness
from rag267.evals.faithfulness import ragas_faithfulness
from rag267.evals.response_relevancy import ragas_response_relevancy
from rag267.evals.answer_accuracy import ragas_answer_accuracy
from rag267.evals.context_relevance import ragas_context_relevance

def main():
    # Parse command-line arguments using argparse
    parser = argparse.ArgumentParser(description="Run a single RAG evaluation")
    parser.add_argument("rag_type", choices=["cohere", "mistral"], help="RAG system type ('cohere' or 'mistral')")
    parser.add_argument("team_type", choices=["engineering", "marketing"], help="Team type ('engineering' or 'marketing')")
    parser.add_argument("top_k", type=int, help="Number of documents to retrieve")
    parser.add_argument("eng_template", help="Path to engineering template")
    parser.add_argument("mkt_template", help="Path to marketing template")
    parser.add_argument("embedding_model", help="Embedding model name")
    parser.add_argument("chunk_size", type=int, help="Chunk size for text splitting")
    parser.add_argument("chunk_overlap", type=int, help="Chunk overlap for text splitting")
    parser.add_argument("--ragas", action="store_true", help="Use only Ragas evaluations (answer_accuracy, context_relevance, faithfulness, response_relevancy)")
    
    args = parser.parse_args()
    
    rag_type = args.rag_type
    team_type = args.team_type
    top_k = args.top_k
    engineering_template = args.eng_template
    marketing_template = args.mkt_template
    embedding_model_name = args.embedding_model
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    use_ragas = args.ragas

    # Get API keys
    cohere_api_key = os.getenv("COHERE_API_KEY_PROD")
    if not cohere_api_key and rag_type == "cohere":
        print("Error: COHERE_API_KEY environment variable not set")
        sys.exit(1)

    logger.info(f"Initializing vector database with {embedding_model_name}, chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    # Initialize vector database manager
    vdm = VectorDatabaseManager(
        embedding_model_name=embedding_model_name,
        collection_name="myrag",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        in_memory=True,
        force_recreate=True,
    )
    
    # Hydrate vector database with data sources from data_sources.py
    logger.info(f"Hydrating vector database with {len(data_sources)} data sources")
    vdm.hydrate(data_sources)

    logger.info(f"Initializing RAG system with {rag_type} model")
    
    # Initialize RAG system
    if rag_type == "cohere":
        rag_system = RAGSystem(
            vector_db_manager=vdm,
            engineering_template_path=engineering_template,
            marketing_template_path=marketing_template,
            cohere_api_key=cohere_api_key,
            use_mistral=False,
            use_cohere=True,
            mistral_model_name=SupportedGeneratorModels.MistralInstructV2.value,
            top_k=top_k,
        )
    else:  # mistral
        rag_system = RAGSystem(
            vector_db_manager=vdm,
            engineering_template_path=engineering_template,
            marketing_template_path=marketing_template,
            cohere_api_key=cohere_api_key,
            use_mistral=True,
            use_cohere=False,
            mistral_model_name=SupportedGeneratorModels.MistralInstructV2.value,
            top_k=top_k,
        )

    # Load validation data and prepare examples
    logger.info("Loading validation data")
    validation_file = os.path.join("data", "validation_question_answers.json")
    with open(validation_file, "r") as f:
        validation_question_answers = json.load(f)

    # Transform the data into LangSmith compatible examples
    examples_engineering = []
    examples_marketing = []

    for sample in validation_question_answers.values():
        examples_engineering.append({
            "inputs": {"question": sample["question"]},
            "outputs": {"answer": sample["gold_answer_research"]}
        })

        examples_marketing.append({
            "inputs": {"question": sample["question"]},
            "outputs": {"answer": sample["gold_answer_marketing"]}
        })

    # Function to get or create dataset
    def get_or_create_dataset(client, dataset_name, examples):
        if client.has_dataset(dataset_name=dataset_name):
            logger.info(f"Dataset '{dataset_name}' already exists, loading existing dataset.")
            dataset = client.read_dataset(dataset_name=dataset_name)
        else:
            logger.info(f"Dataset '{dataset_name}' does not exist, creating it now.")
            dataset = client.create_dataset(dataset_name=dataset_name)
            client.create_examples(dataset_id=dataset.id, examples=examples)
        return dataset

    # Set up evaluation components
    team = Team.Engineering if team_type == "engineering" else Team.Marketing
    dataset_name = f"w267-rag-validation-{team_type}"
    examples = examples_engineering if team_type == "engineering" else examples_marketing
    
    # Create a unique experiment prefix from all parameters
    experiment_prefix = (f"rag-{rag_type}-{team_type}-k{top_k}-"
                        f"emb{embedding_model_name.split('/')[-1]}-"
                        f"cs{chunk_size}-co{chunk_overlap}")
    
    # Modify prefix if using Ragas
    if use_ragas:
        experiment_prefix = f"ragas-{experiment_prefix}"

    logger.info(f"Experiment ID: {experiment_prefix}")

    # Define target function
    def target(inputs: dict) -> dict:
        question = inputs["question"]
        logger.info(f"Processing question: {question[:50]}...")
        answer = rag_system.invoke(team, question)
        retrieved_docs = rag_system.query_vectorstore(question)
        return {
            "answer": answer,
            "documents": retrieved_docs
        }

    # Run evaluation
    client = Client()

    # Get or create the dataset
    dataset = get_or_create_dataset(client, dataset_name, examples)

    # Create results directory
    os.makedirs("results", exist_ok=True)
    result_file = f"results/{experiment_prefix}.json"

    logger.info(f"Starting evaluation: {experiment_prefix}")
    start_time = time.time()
    
    try:
        # Choose which evaluators to use based on the ragas flag
        if use_ragas:
            logger.info("Using Ragas evaluations only")
            evaluators = [
                ragas_answer_accuracy, 
                ragas_context_relevance, 
                ragas_faithfulness, 
                ragas_response_relevancy
            ]
        else:
            logger.info("Using original evaluations")
            evaluators = [
                correctness, 
                groundedness, 
                relevance, 
                retrieval_relevance
            ]
        
        result = client.evaluate(
            target,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=experiment_prefix,
            metadata=rag_system.get_config(),
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Evaluation completed in {elapsed_time:.1f} seconds: {experiment_prefix}")
        return 0
    
    except Exception as e:
        logger.error(f"Error in evaluation {experiment_prefix}: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())