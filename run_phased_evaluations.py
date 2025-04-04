#!/usr/bin/env python
"""
Experimental suite for RAG system evaluation with support for Ragas metrics.
This script runs a series of experiments with different configurations to find optimal settings.

How to Use the Script

Phase 1 (Embedding Model Selection):
python run_experiment_suite.py --phase 1 --max_parallel 2

Phase 2 (Chunking Strategy Selection):
# After analyzing Phase 1 results
python run_experiment_suite.py --phase 2 --embedding_model "multi-qa-mpnet-base-dot-v1" --max_parallel 2

Phase 3 (Retriever Method Selection):
# After analyzing Phase 2 results
python run_experiment_suite.py --phase 3 --embedding_model "multi-qa-mpnet-base-dot-v1" --chunk_size 256 --chunk_overlap 50 --max_parallel 2

To use Ragas metrics instead of custom evaluators, add the --ragas flag:
python run_experiment_suite.py --phase 1 --max_parallel 2 --ragas
"""
import sys
import os
import time
import json
import argparse
import logging
import itertools
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Tuple, Union

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
from rag267.vectordb.utils import Team, SupportedGeneratorModels, SupportedEmbeddingModels
from rag267.vectordb.manager import VectorDatabaseManager
from rag267.data_sources import data_sources

# Standard evaluators
from rag267.evals.correctness import correctness
from rag267.evals.relevance import relevance
from rag267.evals.retrieval_relevance import retrieval_relevance
from rag267.evals.groundedness import groundedness

# Ragas evaluators - Import the fixed versions
from rag267.evals.ragas.faithfulness import ragas_faithfulness
from rag267.evals.ragas.response_relevancy import ragas_response_relevancy
from rag267.evals.ragas.answer_accuracy import ragas_answer_accuracy
from rag267.evals.ragas.context_relevance import ragas_context_relevance

from langsmith import Client


class ExperimentConfig:
    """Configuration for a single experiment"""
    def __init__(
        self,
        rag_type: str,
        team_type: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        retriever_type: str,
        retriever_kwargs: Optional[Dict[str, Any]] = None,
        templates: Optional[Dict[str, str]] = None,
    ):
        self.rag_type = rag_type  # "cohere" or "mistral"
        self.team_type = team_type  # "engineering" or "marketing"
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.retriever_type = retriever_type
        self.retriever_kwargs = retriever_kwargs or {}
        
        # Default templates
        self.templates = {
            "engineering": "templates/engineering_template_3.txt",
            "marketing": "templates/marketing_template_2.txt"
        }
        
        # Override with provided templates if any
        if templates:
            self.templates.update(templates)
            
    def get_experiment_id(self) -> str:
        """Generate a unique experiment ID from the configuration"""
        retriever_extra = ""
        if self.retriever_type != "similarity":
            retriever_extra = f"-{self.retriever_type}"
            if "score_threshold" in self.retriever_kwargs:
                retriever_extra += f"-{self.retriever_kwargs['score_threshold']}"
                
        return (f"rag-{self.rag_type}-{self.team_type}"
                f"-emb-{self.embedding_model.split('/')[-1]}"
                f"-cs{self.chunk_size}-co{self.chunk_overlap}"
                f"-k{self.top_k}{retriever_extra}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            "rag_type": self.rag_type,
            "team_type": self.team_type,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "retriever_type": self.retriever_type,
            "retriever_kwargs": self.retriever_kwargs,
            "templates": self.templates
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary"""
        return cls(
            rag_type=data["rag_type"],
            team_type=data["team_type"],
            embedding_model=data["embedding_model"],
            chunk_size=data["chunk_size"],
            chunk_overlap=data["chunk_overlap"],
            top_k=data["top_k"],
            retriever_type=data["retriever_type"],
            retriever_kwargs=data.get("retriever_kwargs", {}),
            templates=data.get("templates", None)
        )


def create_phase1_experiments() -> List[ExperimentConfig]:
    """Create Phase 1 experiments: Embedding Model Comparison"""
    
    # Define the core parameters to test
    rag_models = ["cohere"] # TODO re-add cohere
    team_types = ["engineering", "marketing"]
    
    # Embedding models to test
    embedding_models = [
        # SupportedEmbeddingModels.MultiQaMpNetBasedDotV1.value,  # Baseline
        # SupportedEmbeddingModels.MpNetBaseV2.value,
        # SupportedEmbeddingModels.MiniLmL6V2.value,
        SupportedEmbeddingModels.DistilRobertaV1.value,
        SupportedEmbeddingModels.MultiQaMpNetBasedCosV1.value
    ]
    
    # Define a focused matrix of experiments
    experiments = []
    
    logger.info("Creating Phase 1 experiments - embedding model comparison")
    for rag_type, team_type, emb_model in itertools.product(rag_models, team_types, embedding_models):
        config = ExperimentConfig(
            rag_type=rag_type,
            team_type=team_type,
            embedding_model=emb_model,
            chunk_size=128,  # Baseline 
            chunk_overlap=0,  # Baseline
            top_k=4,          # Baseline
            retriever_type="similarity"  # Baseline
        )
        experiments.append(config)
    
    logger.info(f"Created {len(experiments)} experiments for Phase 1")
    return experiments


def create_phase2_experiments(best_embedding_model: str) -> List[ExperimentConfig]:
    """Create Phase 2 experiments: Chunk Size and Overlap Optimization"""

    # best performing embedding model from phase 1:
    # marketing: all-mpnet-base-v2
    # engineering: multi-qa-mpnet-base-dot-v1
    
    # Define the core parameters to test
    rag_models = ["cohere"] # TODO optionally add Mistral
    # team_types = ["marketing"] # use for all-mpnet-base-v2
    team_types = ["engineering"] # use for multi-qa-mpnet-base-dot-v1
    
    # Chunk sizes and overlaps to test
    chunk_sizes = [256, 512, 1024]
    chunk_overlaps = [0, 50]
    
    # Define a focused matrix of experiments
    experiments = []
    
    logger.info("Creating Phase 2 experiments - chunk size and overlap test")
    
    for rag_type, team_type, size, overlap in itertools.product(
            rag_models, team_types, chunk_sizes, chunk_overlaps):
        # Skip redundant experiments (baseline already covered in Phase 1)
        if size == 128 and overlap == 0:
            continue
            
        config = ExperimentConfig(
            rag_type=rag_type,
            team_type=team_type,
            embedding_model=best_embedding_model,
            chunk_size=size,
            chunk_overlap=overlap,
            top_k=4,  # Baseline
            retriever_type="similarity"  # Baseline
        )
        experiments.append(config)
    
    logger.info(f"Created {len(experiments)} experiments for Phase 2")
    return experiments


def create_phase3_experiments(best_embedding_model: str, best_chunk_size: int, 
                             best_chunk_overlap: int) -> List[ExperimentConfig]:
    """Create Phase 3 experiments: Retriever Method Optimization"""
    
    # Define the core parameters to test
    rag_models = ["cohere", "mistral"]
    team_types = ["engineering", "marketing"]
    
    # Define a focused matrix of experiments
    experiments = []
    
    logger.info("Creating Phase 3 experiments - retriever type comparison")
    
    # Test different top_k values with similarity search
    for rag_type, team_type, k in itertools.product(rag_models, team_types, [5, 10]):
        config = ExperimentConfig(
            rag_type=rag_type,
            team_type=team_type, 
            embedding_model=best_embedding_model,
            chunk_size=best_chunk_size,
            chunk_overlap=best_chunk_overlap,
            top_k=k,
            retriever_type="similarity"
        )
        experiments.append(config)
    
    # Test similarity search with score thresholds
    for rag_type, team_type, threshold in itertools.product(rag_models, team_types, [0.5, 0.8]):
        config = ExperimentConfig(
            rag_type=rag_type,
            team_type=team_type,
            embedding_model=best_embedding_model,
            chunk_size=best_chunk_size,
            chunk_overlap=best_chunk_overlap,
            top_k=4,  # Baseline
            retriever_type="similarity_score_threshold",
            retriever_kwargs={"score_threshold": threshold}
        )
        experiments.append(config)
    
    # Test MMR retriever
    for rag_type, team_type in itertools.product(rag_models, team_types):
        config = ExperimentConfig(
            rag_type=rag_type,
            team_type=team_type,
            embedding_model=best_embedding_model,
            chunk_size=best_chunk_size,
            chunk_overlap=best_chunk_overlap,
            top_k=4,  # Baseline
            retriever_type="mmr",
            retriever_kwargs={"k": 4, "fetch_k": 10}  # Fetch more but return top 4
        )
        experiments.append(config)
    
    # Test multi-query retriever (one per LLM type)
    for rag_type, team_type in itertools.product(rag_models, team_types):
        # Use the same LLM for query generation as for the main RAG
        config = ExperimentConfig(
            rag_type=rag_type,
            team_type=team_type,
            embedding_model=best_embedding_model,
            chunk_size=best_chunk_size,
            chunk_overlap=best_chunk_overlap,
            top_k=4,  # Baseline
            retriever_type="multi_query",
            retriever_kwargs={"llm_for_queries": rag_type, "k": 4}
        )
        experiments.append(config)
    
    logger.info(f"Created {len(experiments)} experiments for Phase 3")
    return experiments


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


def initialize_rag_system(config: ExperimentConfig, vdm: VectorDatabaseManager, cohere_api_key: str) -> RAGSystem:
    """Initialize a RAG system with the given configuration."""
    logger.info(f"Initializing {config.rag_type} RAG system with {config.retriever_type} retriever")
    
    use_cohere = config.rag_type == "cohere"
    use_mistral = config.rag_type == "mistral"
    
    return RAGSystem(
        vector_db_manager=vdm,
        engineering_template_path=config.templates["engineering"],
        marketing_template_path=config.templates["marketing"],
        cohere_api_key=cohere_api_key,
        use_mistral=use_mistral,
        use_cohere=use_cohere,
        mistral_model_name=SupportedGeneratorModels.MistralInstructV2.value,
        top_k=config.top_k,
        retriever_type=config.retriever_type,
        retriever_kwargs=config.retriever_kwargs,
    )


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


def run_evaluation(config: ExperimentConfig, cohere_api_key: str, use_ragas: bool = False) -> Dict[str, Any]:
    """Run a single evaluation with given experiment configuration."""
    import gc
    import torch
    
    # Initialize vector database manager
    vdm = VectorDatabaseManager(
        embedding_model_name=config.embedding_model,
        collection_name="myrag",
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        in_memory=True,
        force_recreate=True,
    )
    
    # Hydrate vector database with data sources
    logger.info(f"Hydrating vector database with {len(data_sources)} data sources")
    vdm.hydrate(data_sources)
    
    # Initialize RAG system
    rag_system = initialize_rag_system(config, vdm, cohere_api_key)
    
    # Define team and dataset
    team = Team.Engineering if config.team_type == "engineering" else Team.Marketing
    dataset_name = f"w267-rag-validation-{config.team_type}"
    
    # Create LangSmith client
    client = Client()
    
    # Load validation data
    examples_engineering, examples_marketing = load_validation_data()
    examples = examples_engineering if config.team_type == "engineering" else examples_marketing
    
    # Get or create dataset
    dataset = get_or_create_dataset(client, dataset_name, examples)
    
    # Create target function
    target = create_target_function(rag_system, team)
    
    # Get experiment ID
    experiment_id = config.get_experiment_id()
    if use_ragas:
        experiment_id = f"ragas-{experiment_id}"
    logger.info(f"Starting evaluation: {experiment_id}")
    
    start_time = time.time()
    result_data = {}
    
    try:
        # Choose which evaluators to use based on the ragas flag
        if use_ragas:
            logger.info("Using Ragas evaluations")
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
            experiment_prefix=experiment_id,
            metadata=rag_system.get_config(),
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Evaluation completed in {elapsed_time:.1f} seconds: {experiment_id}")
        
        result_data = {
            "experiment_id": experiment_id,
            "config": config.to_dict(),
            "success": True,
            "elapsed_time": elapsed_time,
            "evaluation_type": "ragas" if use_ragas else "original"
        }
    
    except Exception as e:
        logger.error(f"Error in evaluation {experiment_id}: {e}", exc_info=True)
        
        result_data = {
            "experiment_id": experiment_id,
            "config": config.to_dict(),
            "success": False,
            "error": str(e),
            "evaluation_type": "ragas" if use_ragas else "original"
        }
    
    # Clean up to prevent memory leaks
    if config.rag_type == "mistral" and hasattr(rag_system, "llm") and hasattr(rag_system.llm, "pipeline"):
        logger.info(f"Cleaning up Mistral model for {experiment_id}")
        if hasattr(rag_system.llm.pipeline, "model"):
            # Delete the model to free GPU memory
            del rag_system.llm.pipeline.model
            
        # Delete the pipeline
        del rag_system.llm.pipeline
        
    # Delete the RAG system and vector DB manager
    del rag_system
    del vdm
    
    # Force garbage collection
    gc.collect()
    
    # If using CUDA, clear CUDA cache
    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache")
        torch.cuda.empty_cache()
        
    return result_data


def run_experiment_phase(
    phase: int, 
    max_parallel: int = 2,
    best_params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    use_ragas: bool = False
) -> Dict[str, Any]:
    """Run experiments for a specific phase"""
    
    # Get API keys
    cohere_api_key = os.getenv("COHERE_API_KEY_PROD")
    if not cohere_api_key:
        logger.error("Error: COHERE_API_KEY_PROD environment variable not set")
        raise ValueError("COHERE_API_KEY_PROD environment variable is required")
    
    # Create timestamp-based directory if not provided
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_type = "ragas" if use_ragas else "standard"
        output_dir = f"results/phase{phase}_{eval_type}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create experiments based on phase and best parameters from previous phases
    if phase == 1:
        experiments = create_phase1_experiments()
    elif phase == 2:
        if not best_params or 'embedding_model' not in best_params:
            raise ValueError("Phase 2 requires 'embedding_model' from Phase 1 results")
        experiments = create_phase2_experiments(best_params['embedding_model'])
    elif phase == 3:
        if not best_params or not all(k in best_params for k in ['embedding_model', 'chunk_size', 'chunk_overlap']):
            raise ValueError("Phase 3 requires 'embedding_model', 'chunk_size', and 'chunk_overlap' from Phase 2 results")
        experiments = create_phase3_experiments(
            best_params['embedding_model'], 
            best_params['chunk_size'], 
            best_params['chunk_overlap']
        )
    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 1, 2, or 3.")
    
    # Save the experiment plan
    with open(f"{output_dir}/experiment_plan.json", "w") as f:
        json.dump([config.to_dict() for config in experiments], f, indent=4)
    
    # Run experiments in a strategic order to avoid GPU memory issues
    results = []
    
    # Sort experiments so that Mistral experiments are run separately
    # Group by LLM type
    cohere_experiments = [exp for exp in experiments if exp.rag_type == "cohere"]
    mistral_experiments = [exp for exp in experiments if exp.rag_type == "mistral"]
    
    # Reorganize the experiments to avoid running multiple Mistral models at once
    ordered_experiments = cohere_experiments + mistral_experiments
    
    logger.info(f"Running {len(ordered_experiments)} experiments for Phase {phase} with max_parallel={max_parallel}")
    logger.info(f"Running {len(cohere_experiments)} Cohere experiments, then {len(mistral_experiments)} Mistral experiments")
    
    # Only run one experiment at a time if Mistral is involved - Mistral models can't be parallelized on a single GPU
    actual_parallel = 1 if phase == 1 or len(mistral_experiments) > 0 else max_parallel
    logger.info(f"Setting actual parallelism to {actual_parallel} workers due to GPU memory constraints")
    
    with ThreadPoolExecutor(max_workers=actual_parallel) as executor:
        futures = {}
        
        # Submit all experiments
        for config in ordered_experiments:
            future = executor.submit(run_evaluation, config, cohere_api_key, use_ragas)
            futures[future] = config
            
            # If this is a Mistral model, wait for it to complete before submitting another
            if config.rag_type == "mistral":
                logger.info(f"Waiting for Mistral experiment to complete before submitting another")
                future.result()
                
                # Save intermediate results immediately after each Mistral experiment
                if hasattr(future, 'result') and callable(getattr(future, 'result')):
                    result = future.result()
                    results.append(result)
                    
                    # Save intermediate results after each experiment
                    with open(f"{output_dir}/results.json", "w") as f:
                        json.dump(results, f, indent=4)
        
        # For any non-Mistral models that are still running, collect their results
        remaining_futures = [f for f in futures.keys() if not f.done()]
        for future in tqdm(remaining_futures, desc=f"Running Phase {phase} experiments"):
            result = future.result()
            results.append(result)
            
            # Save intermediate results after each experiment
            with open(f"{output_dir}/results.json", "w") as f:
                json.dump(results, f, indent=4)
    
    # Process results to find best configurations
    successful_results = [r for r in results if r.get("success", False)]
    
    if not successful_results:
        logger.error("No successful experiments to analyze")
        return {"phase": phase, "results": results, "output_dir": output_dir, "success": False}
    
    logger.info(f"Completed {len(successful_results)}/{len(experiments)} experiments successfully")
    logger.info(f"Results saved to {output_dir}/results.json")
    
    return {
        "phase": phase,
        "results": results,
        "output_dir": output_dir,
        "success": True
    }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run RAG experiment suite")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3],
                       help="Experiment phase to run (1, 2, or 3)")
    parser.add_argument("--max_parallel", type=int, default=2, 
                       help="Maximum number of parallel experiments (default: 2)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save results (default: auto-generated)")
    
    # Parameters for Phase 2 and 3
    parser.add_argument("--embedding_model", type=str, default=None,
                       help="Best embedding model from Phase 1 (required for Phase 2 and 3)")
    
    # Parameters for Phase 3
    parser.add_argument("--chunk_size", type=int, default=None,
                       help="Best chunk size from Phase 2 (required for Phase 3)")
    parser.add_argument("--chunk_overlap", type=int, default=None,
                       help="Best chunk overlap from Phase 2 (required for Phase 3)")
    
    # Evaluation selector
    parser.add_argument("--ragas", action="store_true",
                       help="Use Ragas evaluations (answer_accuracy, context_relevance, faithfulness, response_relevancy)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Collect best parameters from previous phases based on command line arguments
    best_params = {}
    
    if args.phase >= 2:
        if not args.embedding_model:
            print("Error: --embedding_model is required for Phase 2 and 3")
            sys.exit(1)
        best_params['embedding_model'] = args.embedding_model
    
    if args.phase >= 3:
        if not args.chunk_size or not args.chunk_overlap:
            print("Error: --chunk_size and --chunk_overlap are required for Phase 3")
            sys.exit(1)
        best_params['chunk_size'] = args.chunk_size
        best_params['chunk_overlap'] = args.chunk_overlap
    
    # Log which evaluation set we're using
    if args.ragas:
        logger.info("Using Ragas evaluations: answer_accuracy, context_relevance, faithfulness, response_relevancy")
    else:
        logger.info("Using standard evaluations: correctness, groundedness, relevance, retrieval_relevance")
    
    try:
        # Run the requested phase
        phase_results = run_experiment_phase(
            args.phase,
            args.max_parallel,
            best_params,
            args.output_dir,
            args.ragas
        )
        
        if phase_results["success"]:
            print(f"\nPhase {args.phase} completed successfully!")
            print(f"Results saved to: {phase_results['output_dir']}")
            print("\nTo analyze results and determine the best configuration, examine the metrics in the results.json file.")
            
            evaluator_type = "ragas" if args.ragas else "standard"
            
            if args.phase == 1:
                print("\nAfter analyzing results, run Phase 2 with:")
                print(f"python run_experiment_suite.py --phase 2 --embedding_model <best_model_from_phase1> {'--ragas' if args.ragas else ''}")
            elif args.phase == 2:
                print("\nAfter analyzing results, run Phase 3 with:")
                print(f"python run_experiment_suite.py --phase 3 --embedding_model <best_model> --chunk_size <best_size> --chunk_overlap <best_overlap> {'--ragas' if args.ragas else ''}")
            
            sys.exit(0)
        else:
            print(f"\nPhase {args.phase} failed to complete successfully.")
            print(f"Check the logs and results in: {phase_results['output_dir']}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error running Phase {args.phase}: {str(e)}")
        sys.exit(1)