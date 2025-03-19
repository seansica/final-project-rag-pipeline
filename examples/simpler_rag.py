# Setup and imports
import torch
import os
import bs4
import json
import numpy as np
import time
from pprint import pprint
import locale

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain.llms import HuggingFacePipeline
from langchain_cohere import ChatCohere
from langchain import PromptTemplate, LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.utils.math import cosine_similarity

from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PubMedLoader

from dotenv import load_dotenv

locale.getpreferredencoding = lambda: "UTF-8"

# API keys
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# 2.1 The Embedding Model
# Using sentence transformers for text embeddings
base_embeddings = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")

# Test embedding
text = "This is a test document."
query_result = base_embeddings.embed_query(text)
print(f"Embedding dimension: {len(query_result)}")

doc_result = base_embeddings.embed_documents(
    ["Germany won the World Cup 4 times.", "This is not a test document."]
)
similarity = cosine_similarity([query_result], doc_result)[0]

# 2.2. Loading and Chunking Texts
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

documents = loader.load()

# Text splitting with chunk size of 128 and no overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=0)
splits = text_splitter.split_documents(documents)
print("Number of splits/chunks: ", str(len(splits)))

# 2.3 Storing Embeddings in Vectorstore (Qdrant)
vectorstore = Qdrant.from_documents(
    splits,
    base_embeddings,
    # location=":memory:",  # Local mode with in-memory storage only
    location="http://localhost:6333",  # Local mode with persistent storage
    collection_name="test",
)
retriever = vectorstore.as_retriever()

# Test retrieval
# query = "What is Chain of Thought doing?"
# docs = vectorstore.similarity_search_by_vector(base_embeddings.embed_query(query))

# 2.4. The LLM Setup
# Mistral LLM setup
from huggingface_hub import notebook_login

notebook_login()

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

llm_mistral_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.float32,
    device_map="auto",
    quantization_config=quantization_config,
)

llm_mistral_tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2"
)

# Create a pipeline
mistral_pipe = pipeline(
    "text-generation",
    model=llm_mistral_model,
    tokenizer=llm_mistral_tokenizer,
    max_new_tokens=1000,
    temperature=0.6,
    top_p=0.95,
    do_sample=True,
    repetition_penalty=1.2,
)
mistral_pipe.model.config.pad_token_id = mistral_pipe.model.config.eos_token_id

# 2.5 Testing LLM in LangChain Chain
# Wrap Hugging Face model for LangChain
mistral_llm_lc = HuggingFacePipeline(pipeline=mistral_pipe)

# Define a template
test_llm_template = """[INST] Give me a two-sentence story about an {object}! [/INST]"""
test_llm_prompt_template = PromptTemplate(
    template=test_llm_template, input_variables=["object"]
)

# Create a chain
test_llm_chain_short = (
    {"object": RunnablePassthrough()} | test_llm_prompt_template | mistral_llm_lc
)

# Test Cohere
cohere_chat_model = ChatCohere(cohere_api_key=COHERE_API_KEY)

test_cohere_llm_chain_short = (
    {"object": RunnablePassthrough()} | test_llm_prompt_template | cohere_chat_model
)

# Add output parser
output_parser = StrOutputParser()

test_cohere_llm_chain_short_formatted = (
    {"object": RunnablePassthrough()}
    | test_llm_prompt_template
    | cohere_chat_model
    | output_parser
)

# 2.6 Setting Up a Simple RAG Chain
# Creating a RAG template
rag_template = """[INST] Answer the question based only on the following context:
{context}

Question: {question}
[/INST]
"""
rag_prompt_template = ChatPromptTemplate.from_template(rag_template)

base_rag_chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | rag_prompt_template
    | mistral_llm_lc
    | output_parser
)


# Format function to combine chunks
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Final RAG Chain
rag_template = """[INST]Please answer the question below only based on the context information provided.\n\nHere is a context:\n{context} \n\nHere is a question: \n{question}.[/INST]"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | mistral_llm_lc
)

# Cohere version
cohere_rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | cohere_chat_model
    | output_parser
)

# 3.1 The Vector Database
qdrant_vectorstore = Qdrant.from_documents(
    splits,
    base_embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="rag_tech_db",
    force_recreate=True,
)

retriever = qdrant_vectorstore.as_retriever()

# 3.2 Data Acquisition, Chunking, and Vectorization
# Chunk settings
CHUNK_SIZE = 128
OVERLAP = 0
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP
)

# Document tracking counter
global_doc_number = 1

# Fetch ArXiv papers
arxiv_numbers = (
    "2005.11401",
    "2104.07567",
    "2104.09864",
    "2105.03011",
    "2106.09685",
    "2203.02155",
    "2211.09260",
    "2211.12561",
    "2212.09741",
    "2305.14314",
    "2305.18290",
    "2306.15595",
    "2309.08872",
    "2309.15217",
    "2310.06825",
    "2310.11511",
    "2311.08377",
    "2312.05708",
    "2401.06532",
    "2401.17268",
    "2402.01306",
    "2402.19473",
    "2406.04744",
    "2312.10997",
    "2410.12812",
    "2410.15944",
    "2404.00657",
)

all_arxiv_pages = []

# Loop through the papers
for identifier in arxiv_numbers:
    # Construct URL using the arXiv unique identifier
    arx_url = f"https://arxiv.org/pdf/{identifier}.pdf"

    # Extract pages from the document and add them to the list of pages
    arx_loader = PyMuPDFLoader(arx_url)
    arx_pages = arx_loader.load()
    for page_num in range(len(arx_pages)):
        page = arx_pages[page_num]
        page.metadata["page_num"] = page_num
        page.metadata["doc_num"] = global_doc_number
        page.metadata["doc_source"] = "ArXiv"
        all_arxiv_pages.append(page)

    global_doc_number += 1

# Split ArXiv docs into chunks
splits = text_splitter.split_documents(all_arxiv_pages)
for idx, text in enumerate(splits):
    splits[idx].metadata["split_id"] = idx

# Add to vector database
qdrant_vectorstore.add_documents(documents=splits)

# Fetch Wikipedia articles
wiki_docs = WikipediaLoader(
    query="Generative Artificial Intelligence", load_max_docs=4
).load()
for idx, text in enumerate(wiki_docs):
    wiki_docs[idx].metadata["doc_num"] = global_doc_number
    wiki_docs[idx].metadata["doc_source"] = "Wikipedia"
    global_doc_number += 1

wiki_splits = text_splitter.split_documents(wiki_docs)
for idx, text in enumerate(wiki_splits):
    wiki_splits[idx].metadata["split_id"] = idx

qdrant_vectorstore.add_documents(documents=wiki_splits)

# More Wikipedia articles
wiki_docs = WikipediaLoader(query="Information Retrieval", load_max_docs=4).load()
for idx, text in enumerate(wiki_docs):
    wiki_docs[idx].metadata["doc_num"] = global_doc_number
    wiki_docs[idx].metadata["doc_source"] = "Wikipedia"
    global_doc_number += 1

wiki_splits = text_splitter.split_documents(wiki_docs)
for idx, text in enumerate(wiki_splits):
    wiki_splits[idx].metadata["split_id"] = idx

qdrant_vectorstore.add_documents(documents=wiki_splits)

# LLM Wikipedia article
wiki_docs = WikipediaLoader(query="Large Language Models", load_max_docs=4).load()
for idx, text in enumerate(wiki_docs):
    wiki_docs[idx].metadata["doc_num"] = global_doc_number
    wiki_docs[idx].metadata["doc_source"] = "Wikipedia"
    global_doc_number += 1

wiki_splits = text_splitter.split_documents(wiki_docs)
for idx, text in enumerate(wiki_splits):
    wiki_splits[idx].metadata["split_id"] = idx

qdrant_vectorstore.add_documents(documents=wiki_splits)

# RAG Wikipedia article
wiki_docs = WikipediaLoader(
    query="Retrieval Augmented Generation", load_max_docs=4
).load()
for idx, text in enumerate(wiki_docs):
    wiki_docs[idx].metadata["doc_num"] = global_doc_number
    wiki_docs[idx].metadata["doc_source"] = "Wikipedia"
    global_doc_number += 1

wiki_splits = text_splitter.split_documents(wiki_docs)
for idx, text in enumerate(wiki_splits):
    wiki_splits[idx].metadata["split_id"] = idx

qdrant_vectorstore.add_documents(documents=wiki_splits)

# Add blog entries
web_loader = WebBaseLoader(
    web_paths=(
        "https://lilianweng.github.io/posts/2020-10-29-odqa/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2018-06-24-attention/",
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

web_documents = web_loader.load()

for idx, text in enumerate(web_documents):
    web_documents[idx].metadata["doc_num"] = global_doc_number
    web_documents[idx].metadata["doc_source"] = "WWW"
    global_doc_number += 1

web_splits = text_splitter.split_documents(web_documents)

for idx, text in enumerate(web_splits):
    web_splits[idx].metadata["split_id"] = idx

qdrant_vectorstore.add_documents(documents=web_splits)
