#!/usr/bin/env python
"""
Script to run a single missing experiment and append the results to an existing results.json file.
"""
import sys
import os
import json
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the experiment runner from the main script
from run_phased_evaluations import ExperimentConfig, run_evaluation

def run_missing_experiment(
    results_file_path: str,
    rag_type: str = "cohere",
    team_type: str = "marketing",
    embedding_model: str = "all-MiniLM-L6-v2",
    chunk_size: int = 128,
    chunk_overlap: int = 0,
    top_k: int = 4,
    use_ragas: bool = True,
    use_deepeval: bool = False,
    limit: int = 78
) -> Dict[str, Any]:
    """Run a single missing experiment and append its results to the existing results file."""
    
    # Verify results file exists
    if not os.path.exists(results_file_path):
        logger.error(f"Results file not found: {results_file_path}")
        sys.exit(1)
    
    # Load existing results
    with open(results_file_path, "r") as f:
        results = json.load(f)
    
    # Create experiment config
    config = ExperimentConfig(
        rag_type=rag_type,
        team_type=team_type,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retriever_type="similarity",
        top_k=top_k,  # For similarity retriever, top_k will be handled internally
    )
    
    # Generate experiment ID - no longer modifying based on evaluation type
    experiment_id = config.get_experiment_id()
    
    # Check if the experiment already exists in results
    for result in results:
        if result.get("experiment_id") == experiment_id:
            logger.warning(f"Experiment {experiment_id} already exists in results")
            return None
    
    # Get API key
    cohere_api_key = os.getenv("COHERE_API_KEY_PROD")
    if not cohere_api_key:
        logger.error("COHERE_API_KEY_PROD environment variable not set")
        sys.exit(1)
    
    # Create args object for the run_evaluation function with proper flags
    class Args:
        def __init__(self):
            self.ragas = use_ragas
            self.deepeval = use_deepeval
            self.bertscore = args.bertscore if hasattr(args, 'bertscore') else False  # add BERTScore support
            self.standard = not (use_ragas or use_deepeval or self.bertscore)  # Use standard if no others specified
            
    args_obj = Args()
    
    # Run the evaluation
    logger.info(f"Running missing experiment: {experiment_id}")
    
    # Log which evaluators are being used
    evaluator_types = []
    if args_obj.standard:
        evaluator_types.append("standard")
    if args_obj.ragas:
        evaluator_types.append("RAGAS")
    if args_obj.deepeval:
        evaluator_types.append("DeepEval")
    if args_obj.bertscore:
        evaluator_types.append("BERTScore")
    
    logger.info(f"Using evaluators: {', '.join(evaluator_types)}")
    
    # Run the evaluation with all selected evaluator types
    result = run_evaluation(config, cohere_api_key, use_ragas=use_ragas, limit=limit, args=args_obj)
    
    # Add the result to the results list
    results.append(result)
    
    # Save the updated results
    with open(results_file_path, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Updated results file with experiment: {experiment_id}")
    
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a single missing experiment and append to results")
    parser.add_argument("--results_file", type=str, required=True, 
                        help="Path to the existing results.json file")
    parser.add_argument("--rag_type", type=str, default="cohere",
                        help="RAG type (default: cohere)")
    parser.add_argument("--team_type", type=str, default="marketing",
                        help="Team type (default: marketing)")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                        help="Embedding model (default: all-MiniLM-L6-v2)")
    parser.add_argument("--chunk_size", type=int, default=128,
                        help="Chunk size (default: 128)")
    parser.add_argument("--chunk_overlap", type=int, default=0,
                        help="Chunk overlap (default: 0)")
    parser.add_argument("--top_k", type=int, default=4,
                        help="Top k (default: 4)")
    parser.add_argument("--limit", type=int, default=78,
                        help="Question limit (default: 78)")
    parser.add_argument("--no_ragas", action="store_true",
                        help="Don't use RAGAS evaluations")
    parser.add_argument("--standard", action="store_true",
                        help="Use standard evaluations (will be used by default if no other evaluators are specified)")
    parser.add_argument("--deepeval", action="store_true",
                        help="Use DeepEval evaluations (faithfulness, geval) - requires OpenAI API key")
    parser.add_argument("--bertscore", action="store_true",
                        help="Use BERTScore evaluation to measure semantic similarity with reference answers")
    
    args = parser.parse_args()
    
    # Run the missing experiment
    run_missing_experiment(
        results_file_path=args.results_file,
        rag_type=args.rag_type,
        team_type=args.team_type,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        use_ragas=not args.no_ragas,
        use_deepeval=args.deepeval,
        limit=args.limit
        # bertscore will be handled directly from args within run_missing_experiment
    )