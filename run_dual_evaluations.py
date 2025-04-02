#!/usr/bin/env python
"""
Script to run multiple RAG evaluations concurrently using threading.
"""
import sys
import os
import time
import json
import threading
import argparse
import logging
from dotenv import load_dotenv
from langsmith import Client
from typing import Dict, List, Any, Tuple

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import custom modules
from rag267.rag import RAGSystem
from rag267.vectordb.utils import Team, SupportedGeneratorModels
from rag267.vectordb.manager import VectorDatabaseManager
from rag267.data_sources import data_sources
from rag267.evals.correctness import correctness
from rag267.evals.relevance import relevance
from rag267.evals.retrieval_relevance import retrieval_relevance
from rag267.evals.groundedness import groundedness


def initialize_rag_system(rag_type: str, vdm: VectorDatabaseManager, 
                        engineering_template: str, marketing_template: str,
                        cohere_api_key: str, top_k: int) -> RAGSystem:
    """Initialize a RAG system with the given parameters."""
    logger.info(f"Initializing {rag_type} RAG system")
    
    if rag_type == "cohere":
        return RAGSystem(
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
        return RAGSystem(
            vector_db_manager=vdm,
            engineering_template_path=engineering_template,
            marketing_template_path=marketing_template,
            cohere_api_key=cohere_api_key,
            use_mistral=True,
            use_cohere=False,
            mistral_model_name=SupportedGeneratorModels.MistralInstructV2.value,
            top_k=top_k,
        )


def load_validation_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load validation data and prepare examples for both teams."""
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
    
    return examples_engineering, examples_marketing


def get_or_create_dataset(client: Client, dataset_name: str, examples: List[Dict[str, Any]]):
    """Get or create a LangSmith dataset."""
    if client.has_dataset(dataset_name=dataset_name):
        logger.info(f"Dataset '{dataset_name}' already exists, loading existing dataset.")
        dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        logger.info(f"Dataset '{dataset_name}' does not exist, creating it now.")
        dataset = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(dataset_id=dataset.id, examples=examples)
    return dataset


def create_target_function(rag_system: RAGSystem, team: Team):
    """Create a target function for evaluation with a specific rag system and team."""
    def target(inputs: dict) -> dict:
        question = inputs["question"]
        logger.info(f"Processing question: {question[:50]}...")
        answer = rag_system.invoke(team, question)
        retrieved_docs = rag_system.query_vectorstore(question)
        return {
            "answer": answer,
            "documents": retrieved_docs
        }
    return target


def run_evaluation(rag_type: str, team_type: str, rag_system: RAGSystem, 
                  dataset_name: str, experiment_prefix: str, 
                  embedding_model_name: str, chunk_size: int, 
                  chunk_overlap: int, top_k: int):
    """Run a single evaluation with given parameters."""
    client = Client()
    
    # Define team
    team = Team.Engineering if team_type == "engineering" else Team.Marketing
    
    # Create target function
    target = create_target_function(rag_system, team)
    
    logger.info(f"Starting evaluation: {experiment_prefix}")
    start_time = time.time()
    
    try:
        result = client.evaluate(
            target,
            data=dataset_name,
            evaluators=[correctness, groundedness, relevance, retrieval_relevance],
            experiment_prefix=experiment_prefix,
            metadata=rag_system.get_config(),
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Evaluation completed in {elapsed_time:.1f} seconds: {experiment_prefix}")
        return 0
    
    except Exception as e:
        logger.error(f"Error in evaluation {experiment_prefix}: {e}", exc_info=True)
        return 1


def run_concurrent_evaluations(args):
    """Run evaluations concurrently using threading."""
    # Get API keys
    cohere_api_key = os.getenv("COHERE_API_KEY_PROD")
    if not cohere_api_key:
        logger.error("Error: COHERE_API_KEY_PROD environment variable not set")
        return 1

    # Initialize shared vector database manager
    vdm = VectorDatabaseManager(
        embedding_model_name=args.embedding_model,
        collection_name="myrag",
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        in_memory=True,
        force_recreate=True,
    )
    
    # Hydrate vector database with data sources from data_sources.py
    logger.info(f"Hydrating vector database with {len(data_sources)} data sources")
    vdm.hydrate(data_sources)

    # Initialize RAG systems with shared vector database
    rag_system_cohere = initialize_rag_system(
        "cohere", vdm, args.eng_template, args.mkt_template, 
        cohere_api_key, args.top_k
    )
    rag_system_mistral = initialize_rag_system(
        "mistral", vdm, args.eng_template, args.mkt_template, 
        cohere_api_key, args.top_k
    )

    # Load validation data
    examples_engineering, examples_marketing = load_validation_data()

    # Create LangSmith client and datasets
    client = Client()
    dataset_eng = get_or_create_dataset(client, "w267-rag-validation-engineering", examples_engineering)
    dataset_mkt = get_or_create_dataset(client, "w267-rag-validation-marketing", examples_marketing)

    # Define all evaluation configurations
    # Format: (rag_type, team_type, rag_system, dataset_name)
    eval_configs = [
        # First phase: Cohere+Eng and Mistral+Mkt
        ("cohere", "engineering", rag_system_cohere, "w267-rag-validation-engineering"),
        ("mistral", "marketing", rag_system_mistral, "w267-rag-validation-marketing"),
        # Second phase: Cohere+Mkt and Mistral+Eng
        ("cohere", "marketing", rag_system_cohere, "w267-rag-validation-marketing"),
        ("mistral", "engineering", rag_system_mistral, "w267-rag-validation-engineering")
    ]

    # Run evaluations in two phases
    for phase in range(2):
        threads = []
        logger.info(f"Starting evaluation phase {phase+1}/2")
        
        # Create threads for this phase (2 at a time)
        for i in range(2):
            idx = phase * 2 + i
            rag_type, team_type, rag_system, dataset_name = eval_configs[idx]
            
            # Create a unique experiment prefix
            experiment_prefix = (f"rag-{rag_type}-{team_type}-k{args.top_k}-"
                                f"emb{args.embedding_model.split('/')[-1]}-"
                                f"cs{args.chunk_size}-co{args.chunk_overlap}")
            
            # Create and start the thread
            thread = threading.Thread(
                target=run_evaluation,
                args=(rag_type, team_type, rag_system, dataset_name, 
                      experiment_prefix, args.embedding_model, 
                      args.chunk_size, args.chunk_overlap, args.top_k)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads in this phase to complete
        for thread in threads:
            thread.join()
        
        logger.info(f"Completed evaluation phase {phase+1}/2")

    logger.info("All evaluations completed successfully")
    return 0


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run concurrent RAG evaluations")
    parser.add_argument("--top_k", type=int, default=4, help="Number of documents to retrieve")
    parser.add_argument("--eng_template", type=str, default="templates/engineering_template.txt", 
                        help="Path to engineering template")
    parser.add_argument("--mkt_template", type=str, default="templates/marketing_template.txt", 
                        help="Path to marketing template")
    parser.add_argument("--embedding_model", type=str, default="multi-qa-mpnet-base-dot-v1", 
                        help="Embedding model name")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for text splitting")
    parser.add_argument("--chunk_overlap", type=int, default=0, help="Chunk overlap for text splitting")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(run_concurrent_evaluations(args))