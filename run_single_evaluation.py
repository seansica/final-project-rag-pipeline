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
import re
import json
import os
import time
import argparse
import numpy as np
from dotenv import load_dotenv
from langsmith import Client
from langchain_openai import ChatOpenAI
from typing_extensions import Annotated, TypedDict
from typing import Dict, List, Any, Optional
import logging
import torch
import gc

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import custom modules (using absolute imports)
from src.rag267.rag import RAGSystem
from src.rag267.vectordb.utils import Team, SupportedGeneratorModels
from src.rag267.vectordb.manager import VectorDatabaseManager
from src.rag267.data_sources import data_sources

# Standard evaluators
from src.rag267.evals.correctness import correctness
from src.rag267.evals.relevance import relevance
from src.rag267.evals.retrieval_relevance import retrieval_relevance
from src.rag267.evals.groundedness import groundedness

# Ragas evaluators
from src.rag267.evals.ragas.faithfulness import ragas_faithfulness
from src.rag267.evals.ragas.response_relevancy import ragas_response_relevancy
from src.rag267.evals.ragas.answer_accuracy import ragas_answer_accuracy
from src.rag267.evals.ragas.context_relevance import ragas_context_relevance

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
    parser.add_argument("--retriever_type", default="similarity", choices=["similarity", "similarity_score_threshold", "mmr", "multi_query"], 
                       help="Type of retriever to use (default: similarity)")
    parser.add_argument("--retriever_kwargs", type=str, help="JSON string of retriever kwargs (e.g. '{\"score_threshold\": 0.8}')")
    parser.add_argument("--limit", type=int, default=78, help="Limit the number of questions used in evaluation (default: 78)")
    parser.add_argument("--output_dir", type=str, help="Directory to save outputs (default: results/single_evaluations)")
    
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
    retriever_type = args.retriever_type
    retriever_kwargs = json.loads(args.retriever_kwargs) if args.retriever_kwargs else {}
    limit = args.limit
    
    # Set up output directory
    output_dir = args.output_dir or "results/single_evaluations"
    os.makedirs(output_dir, exist_ok=True)

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
    use_cohere = rag_type == "cohere"
    use_mistral = rag_type == "mistral"
    
    logger.info(f"Initializing RAG system with: {retriever_type} retriever, top_k={top_k}")
    if retriever_kwargs:
        logger.info(f"Retriever kwargs: {retriever_kwargs}")
    
    rag_system = RAGSystem(
        vector_db_manager=vdm,
        engineering_template_path=engineering_template,
        marketing_template_path=marketing_template,
        cohere_api_key=cohere_api_key,
        use_mistral=use_mistral,
        use_cohere=use_cohere,
        mistral_model_name=SupportedGeneratorModels.MistralInstructV2.value,
        top_k=top_k,
        retriever_type=retriever_type,
        retriever_kwargs=retriever_kwargs,
    )

    # Load validation data and prepare examples
    logger.info("Loading validation data")
    validation_file = os.path.join("data", "validation_question_answers.json")
    with open(validation_file, "r") as f:
        validation_question_answers = json.load(f)

    # Transform the data into LangSmith compatible examples
    examples_engineering = []
    examples_marketing = []

    # Apply limit to the validation data
    counter = 0
    for sample in validation_question_answers.values():
        if counter >= limit:
            break
            
        examples_engineering.append({
            "inputs": {"question": sample["question"]},
            "outputs": {"answer": sample["gold_answer_research"]}
        })

        examples_marketing.append({
            "inputs": {"question": sample["question"]},
            "outputs": {"answer": sample["gold_answer_marketing"]}
        })
        
        counter += 1
        
    logger.info(f"Loaded {counter} validation questions (limit={limit})")

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
    
    # If using a limit, append it to the dataset name
    if limit < 78:
        dataset_name = f"{dataset_name}-limit{limit}"
        
    examples = examples_engineering if team_type == "engineering" else examples_marketing
    
    # Create a unique experiment prefix from all parameters
    experiment_prefix = (f"rag-{rag_type}-{team_type}-"
                        f"emb-{embedding_model_name.split('/')[-1]}-"
                        f"cs{chunk_size}-co{chunk_overlap}-"
                        f"k{top_k}")
                        
    # Add retriever info if not using default similarity
    if retriever_type != "similarity":
        retriever_extra = f"-{retriever_type}"
        if "score_threshold" in retriever_kwargs:
            retriever_extra += f"-{retriever_kwargs['score_threshold']}"
        experiment_prefix += retriever_extra
    
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

    # Create experiment directory within output_dir
    experiment_dir = os.path.join(output_dir, experiment_prefix)
    os.makedirs(experiment_dir, exist_ok=True)
    result_file = os.path.join(experiment_dir, "results.json")

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
        
        # Run the evaluation
        result = client.evaluate(
            target,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=experiment_prefix,
            metadata=rag_system.get_config(),
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Evaluation completed in {elapsed_time:.1f} seconds: {experiment_prefix}")
        
        # Convert result to pandas dataframe to compute summary statistics
        result_df = result.to_pandas()
        
        # Compute summary statistics
        feedback_metrics = {}
        
        # Process feedback metrics
        feedback_cols = [col for col in result_df.columns if col.startswith('feedback.')]
        for col in feedback_cols:
            metric_name = col.replace('feedback.', '')
            values = result_df[col].dropna()
            
            if len(values) == 0:
                continue
                
            feedback_metrics[metric_name] = {
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'count': int(len(values))
            }
        
        # Compare output answers with reference answers
        text_stats = {}
        if 'outputs.answer' in result_df.columns and 'reference.answer' in result_df.columns:
            # Character length comparison
            output_char_lens = result_df['outputs.answer'].fillna('').astype(str).apply(len)
            reference_char_lens = result_df['reference.answer'].fillna('').astype(str).apply(len)
            char_ratios = output_char_lens / reference_char_lens.replace(0, float('nan'))
            
            # Word count comparison (simple word tokenization)
            output_word_counts = result_df['outputs.answer'].fillna('').astype(str).apply(
                lambda x: len(re.findall(r'\b\w+\b', x.lower()))
            )
            reference_word_counts = result_df['reference.answer'].fillna('').astype(str).apply(
                lambda x: len(re.findall(r'\b\w+\b', x.lower()))
            )
            word_ratios = output_word_counts / reference_word_counts.replace(0, float('nan'))
            
            text_stats = {
                'character_length': {
                    'mean_output': float(output_char_lens.mean()),
                    'mean_reference': float(reference_char_lens.mean()),
                    'mean_ratio': float(char_ratios.mean())
                },
                'word_count': {
                    'mean_output': float(output_word_counts.mean()),
                    'mean_reference': float(reference_word_counts.mean()),
                    'mean_ratio': float(word_ratios.mean())
                }
            }
            
            # Calculate TF-IDF similarity if there are enough examples
            if len(result_df) >= 2:
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.metrics.pairwise import cosine_similarity
                    
                    outputs = result_df['outputs.answer'].fillna('').astype(str).tolist()
                    references = result_df['reference.answer'].fillna('').astype(str).tolist()
                    
                    vectorizer = TfidfVectorizer()
                    all_docs = outputs + references
                    tfidf_matrix = vectorizer.fit_transform(all_docs)
                    
                    n = len(outputs)
                    similarities = []
                    for i in range(n):
                        if i < len(tfidf_matrix) and i + n < len(tfidf_matrix):
                            output_vector = tfidf_matrix[i]
                            reference_vector = tfidf_matrix[i + n]
                            similarity = cosine_similarity(output_vector, reference_vector)[0][0]
                            similarities.append(similarity)
                    
                    if similarities:
                        text_stats['tfidf_similarity'] = {
                            'mean': float(np.mean(similarities)),
                            'min': float(np.min(similarities)),
                            'max': float(np.max(similarities))
                        }
                except Exception as e:
                    logger.warning(f"Could not compute TF-IDF similarity: {e}")
        
        # Create final result data
        result_data = {
            "experiment_id": experiment_prefix,
            "config": {
                "rag_type": rag_type,
                "team_type": team_type,
                "embedding_model": embedding_model_name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k": top_k,
                "retriever_type": retriever_type,
                "retriever_kwargs": retriever_kwargs,
            },
            "success": True,
            "elapsed_time": elapsed_time,
            "evaluation_type": "ragas" if use_ragas else "original",
            "metrics": {
                "feedback": feedback_metrics,
                "text_comparison": text_stats
            }
        }
        
        # Save the results
        with open(result_file, 'w') as f:
            # Use a JSON encoder that can handle NumPy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super().default(obj)
                    
            json.dump(result_data, f, indent=2, cls=NumpyEncoder)
            
        logger.info(f"Results saved to {result_file}")
        
        # Clean up resources
        if use_ragas:
            try:
                from src.rag267.evals.ragas import embedding_model
                embedding_model.clear_embeddings()
                logger.info("Cleaned up RAGAS embedding model")
            except Exception as e:
                logger.warning(f"Failed to clear RAGAS embedding model: {e}")
                
        # Force garbage collection
        gc.collect()
        
        # If using CUDA, clear CUDA cache
        if torch.cuda.is_available():
            logger.info("Clearing CUDA cache")
            torch.cuda.empty_cache()
            
        return 0
    
    except Exception as e:
        logger.error(f"Error in evaluation {experiment_prefix}: {e}", exc_info=True)
        
        # Clean up resources on error
        if use_ragas and 'embedding_model' in sys.modules.get('src.rag267.evals.ragas', {}).__dict__:
            try:
                from src.rag267.evals.ragas import embedding_model
                embedding_model.clear_embeddings()
            except Exception:
                pass
                
        # Force garbage collection
        gc.collect()
        
        # If using CUDA, clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return 1

if __name__ == "__main__":
    sys.exit(main())