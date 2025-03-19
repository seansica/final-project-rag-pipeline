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
