# #!/usr/bin/env python
# """
# Script to run a single RAG evaluation with specific parameters.
# Usage: python run_single_evaluation.py cohere engineering 4
# """

# import sys
# import json
# import os
# from typing import Dict, List, Optional, Any
# import enum
# import torch
# from loguru import logger
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     pipeline,
#     BitsAndBytesConfig,
# )
# from langchain_huggingface import HuggingFacePipeline
# from langchain_cohere import ChatCohere
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_community.vectorstores import Qdrant
# from langsmith import Client
# from langchain_openai import ChatOpenAI
# from typing_extensions import Annotated, TypedDict

# # Import custom modules
# from ..rag import RAGSystem
# from ..vectordb.utils import Team, SupportedGeneratorModels
# from ..vectordb.manager import VectorDatabaseManager
# from .correctness import correctness
# from .relevance import relevance
# from .retrieval_relevance import retrieval_relevance


# # Parse command-line arguments
# if len(sys.argv) != 8:
#     print("Usage: python run_single_evaluation.py <rag_type> <team_type> <top_k> <eng_template> <mkt_template> <embedding_model> <chunk_size> <chunk_overlap>")
#     sys.exit(1)

# rag_type = sys.argv[1]  # 'cohere' or 'mistral'
# team_type = sys.argv[2]  # 'engineering' or 'marketing'
# top_k = int(sys.argv[3])
# engineering_template = sys.argv[4]
# marketing_template = sys.argv[5]
# embedding_model_name = sys.argv[6]
# chunk_size = int(sys.argv[7])
# chunk_overlap = int(sys.argv[8])

# # Get API keys
# cohere_api_key = os.getenv("COHERE_API_KEY")
# if not cohere_api_key and rag_type == "cohere":
#     print("Error: COHERE_API_KEY environment variable not set")
#     sys.exit(1)

# # Initialize vector database manager
# vdm = VectorDatabaseManager(
#     embedding_model_name=embedding_model_name,  # "multi-qa-mpnet-base-dot-v1",
#     collection_name="myrag",
#     chunk_size=chunk_size,  # 128,
#     chunk_overlap=chunk_overlap,  # 0,
#     in_memory=True,
#     force_recreate=True,
# )

# # Initialize RAG system
# if rag_type == "cohere":
#     rag_system = RAGSystem(
#         vector_db_manager=vdm,
#         engineering_template_path=engineering_template,
#         marketing_template_path=marketing_template,
#         cohere_api_key=cohere_api_key,
#         use_mistral=False,
#         use_cohere=True,
#         mistral_model_name=SupportedGeneratorModels.MistralInstructV2.value,
#         top_k=top_k,
#     )
# else:  # mistral
#     rag_system = RAGSystem(
#         vector_db_manager=vdm,
#         engineering_template_path=engineering_template,
#         marketing_template_path=marketing_template,
#         cohere_api_key=cohere_api_key,
#         use_mistral=True,
#         use_cohere=False,
#         mistral_model_name=SupportedGeneratorModels.MistralInstructV2.value,
#         top_k=top_k,
#     )

# # Load validation data and prepare examples
# with open("../data/validation_question_answers.json", "r") as f:
#     validation_question_answers = json.load(f)

# # Transform the data into LangSmith compatible examples
# examples_engineering = []
# examples_marketing = []

# for sample in validation_question_answers.values():
#     examples_engineering.append(
#         {
#             "inputs": {"question": sample["question"]},
#             "outputs": {"answer": sample["gold_answer_research"]},
#         }
#     )

#     examples_marketing.append(
#         {
#             "inputs": {"question": sample["question"]},
#             "outputs": {"answer": sample["gold_answer_marketing"]},
#         }
#     )


# # Function to get or create dataset
# def get_or_create_dataset(client, dataset_name, examples):
#     if client.has_dataset(dataset_name=dataset_name):
#         print(f"Dataset '{dataset_name}' already exists, loading existing dataset.")
#         dataset = client.read_dataset(dataset_name=dataset_name)
#     else:
#         print(f"Dataset '{dataset_name}' does not exist, creating it now.")
#         dataset = client.create_dataset(dataset_name=dataset_name)
#         client.create_examples(dataset_id=dataset.id, examples=examples)
#     return dataset


# # Set up evaluation components
# team = Team.Engineering if team_type == "engineering" else Team.Marketing
# dataset_name = f"w267-rag-validation-{team_type}"
# examples = examples_engineering if team_type == "engineering" else examples_marketing
# experiment_prefix = f"rag-{rag_type}-eval-{team_type}-k{top_k}-{engineering_template}-{marketing_template}"


# # Define target function
# def target(inputs: dict) -> dict:
#     question = inputs["question"]
#     answer = rag_system.invoke(team, question)
#     retrieved_docs = rag_system.query_vectorstore(question)
#     return {"answer": answer, "documents": retrieved_docs}


# # Run evaluation
# client = Client()

# # Get or create the dataset
# dataset = get_or_create_dataset(client, dataset_name, examples)

# logger.info(f"Starting evaluation: {experiment_prefix}")
# try:
#     result = client.evaluate(
#         target,
#         data=dataset_name,
#         evaluators=[correctness, groundedness, relevance, retrieval_relevance],
#         experiment_prefix=experiment_prefix,
#         metadata=rag_system.get_config(),
#     )

#     # Save result to file
#     os.makedirs("results", exist_ok=True)
#     with open(f"results/{experiment_prefix}.json", "w") as f:
#         json.dump(result, f)

#     logger.info(f"Evaluation completed: {experiment_prefix}")
# except Exception as e:
#     logger.info(f"Error in evaluation {experiment_prefix}: {e}")
#     sys.exit(1)
