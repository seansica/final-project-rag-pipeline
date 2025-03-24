import bs4
import enum
from loguru import logger
from typing import List, Any, Dict, Optional
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    WikipediaLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document


class SupportedGeneratorModels(enum.Enum):
    MistralInstructV2 = "mistralai/Mistral-7B-Instruct-v0.2"

class ModelType(enum.Enum):
    Mistral = "mistral"
    Cohere = "cohere"

class Team(enum.Enum):
    Engineering = "engineering"
    Marketing = "marketing"

class SupportedEmbeddingModels(enum.Enum):
    MpNetBaseV2 = "all-mpnet-base-v2"
    MiniLmL6V2 = "all-MiniLM-L6-v2"
    DistilRobertaV1 = "all-distilroberta-v1"
    MultiQaMpNetBasedCosV1 = "multi-qa-mpnet-base-cos-v1"
    MultiQaMpNetBasedDotV1 = "multi-qa-mpnet-base-dot-v1"


class SourceType(enum.Enum):
    ARXIV = "arxiv"
    WIKIPEDIA = "wikipedia"
    WEBSITE = "website"


@dataclass
class DataSource:
    identifier: str
    source_type: SourceType

    # Optional metadata that can be added to all documents from this source
    additional_metadata: Optional[Dict[str, Any]] = None


def get_embedding_model(model: str) -> HuggingFaceEmbeddings:
    """
    Creates and returns a HuggingFaceEmbeddings model for text embeddings.

    Args:
        model (str): The name of the Hugging Face model to use for embeddings.
                    Should be one of the values in SupportedEmbeddingModels.

    Returns:
        HuggingFaceEmbeddings: The initialized embedding model.

    Raises:
        ValueError: If the model name is empty, invalid, or not supported.
        ImportError: If the required dependencies are not installed.
    """
    if not model or not isinstance(model, str):
        raise ValueError("Model name must be a non-empty string")

    # Check if model is supported
    supported_models = [m.value for m in SupportedEmbeddingModels]
    if model not in supported_models:
        raise ValueError(
            f"Unsupported model: {model}. Must be one of: {supported_models}"
        )

    try:
        return HuggingFaceEmbeddings(model_name=model)
    except ImportError as e:
        raise ImportError(f"Missing dependencies for HuggingFaceEmbeddings: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to load embedding model '{model}': {str(e)}")


def initialize_vector_db(
    embedding_model_name: str = SupportedEmbeddingModels.MultiQaMpNetBasedCosV1,
    collection_name: str = "rag267",
    in_memory: bool = True,
    force_recreate: bool = True,
) -> Qdrant:
    """
    Initialize a Qdrant vector database with specified embedding model.

    Args:
        embedding_model_name: Name of the Hugging Face embedding model to use
        collection_name: Name for the Qdrant collection
        in_memory: Whether to use in-memory storage (True) or disk storage (False)
        force_recreate: Whether to recreate the collection if it exists

    Returns:
        Initialized Qdrant vectorstore object
    """
    # Initialize embeddings
    embeddings = get_embedding_model(embedding_model_name)

    # Location for Qdrant storage
    db_location = (
        ":memory:" if in_memory else "http://localhost:6333"
    )  # TODO make this more robust

    # Create an empty Qdrant vectorstore
    vectorstore = Qdrant.from_documents(
        documents=[
            Document(page_content="Initialization document", metadata={})
        ],  # Just to initialize
        embedding=embeddings,
        location=db_location,
        collection_name=collection_name,
        force_recreate=force_recreate,
    )

    return vectorstore


def load_arxiv_documents(
    arxiv_id: str, doc_id: int, additional_metadata: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    Load documents from an ArXiv paper.

    Args:
        arxiv_id: ArXiv ID of the paper
        doc_id: Document ID to assign
        additional_metadata: Additional metadata to add to documents

    Returns:
        List of Document objects
    """
    # Construct URL using the arXiv unique identifier
    arx_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    try:
        # Extract pages from the document
        arx_loader = PyMuPDFLoader(arx_url)
        arx_pages = arx_loader.load()

        # Add metadata to each page
        for page_num in range(len(arx_pages)):
            page = arx_pages[page_num]
            page.metadata["page_num"] = page_num
            page.metadata["doc_num"] = doc_id
            page.metadata["doc_source"] = "ArXiv"
            page.metadata["source_id"] = arxiv_id

            # Add any additional metadata
            if additional_metadata:
                for key, value in additional_metadata.items():
                    page.metadata[key] = value

        return arx_pages
    except Exception as e:
        logger.exception(f"Error loading ArXiv document {arxiv_id}: {e}")
        return []


def load_wikipedia_documents(
    query: str,
    doc_id: int,
    max_docs: int = 4,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Load documents from Wikipedia.

    Args:
        query: Search query for Wikipedia
        doc_id: Document ID to assign
        max_docs: Maximum number of documents to load
        additional_metadata: Additional metadata to add to documents

    Returns:
        List of Document objects
    """
    try:
        wiki_docs = WikipediaLoader(query=query, load_max_docs=max_docs).load()

        # Add metadata to each document
        for doc in wiki_docs:
            doc.metadata["doc_num"] = doc_id
            doc.metadata["doc_source"] = "Wikipedia"
            doc.metadata["source_id"] = query

            # Add any additional metadata
            if additional_metadata:
                for key, value in additional_metadata.items():
                    doc.metadata[key] = value

        return wiki_docs
    except Exception as e:
        logger.exception(f"Error loading Wikipedia documents for query {query}: {e}")
        return []


def load_website_documents(
    url: str, doc_id: int, additional_metadata: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    Load documents from a website.

    Args:
        url: URL of the website
        doc_id: Document ID to assign
        additional_metadata: Additional metadata to add to documents

    Returns:
        List of Document objects
    """
    try:
        # Set up the web loader with BeautifulSoup settings to extract relevant content
        web_loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=(
                        "post-content",
                        "post-title",
                        "post-header",
                        "article",
                        "content",
                        "main",
                    )
                )
            ),
        )

        web_documents = web_loader.load()

        # Add metadata to each document
        for doc in web_documents:
            doc.metadata["doc_num"] = doc_id
            doc.metadata["doc_source"] = "Website"
            doc.metadata["source_id"] = url

            # Add any additional metadata
            if additional_metadata:
                for key, value in additional_metadata.items():
                    doc.metadata[key] = value

        return web_documents
    except Exception as e:
        logger.exception(f"Error loading website document from {url}: {e}")
        return []


def split_and_add_documents(
    vectorstore: Qdrant,
    documents: List[Document],
    chunk_size: int = 128,
    chunk_overlap: int = 0,
) -> None:
    """
    Split documents into chunks and add them to the vectorstore.

    Args:
        vectorstore: Qdrant vectorstore object
        documents: List of Document objects
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
    """
    if not documents:
        return

    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split documents
    splits = text_splitter.split_documents(documents)

    # Add split_id to metadata
    for idx, split in enumerate(splits):
        split.metadata["split_id"] = idx

    # Add to vectorstore
    vectorstore.add_documents(documents=splits)

    logger.info(f"Added {len(splits)} chunks to the vectorstore")


def hydrate_vector_db(
    vectorstore: Qdrant,
    data_sources: List[DataSource],
    chunk_size: int = 128,
    chunk_overlap: int = 0,
) -> None:
    """
    Hydrate the vector database with documents from various sources.

    Args:
        vectorstore: Qdrant vectorstore object
        data_sources: List of DataSource objects
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
    """
    doc_id = 1

    for data_source in data_sources:
        logger.info(
            f"Processing source: {data_source.identifier} ({data_source.source_type.value})"
        )

        documents = []

        # Route to the appropriate loading function based on source type
        if data_source.source_type == SourceType.ARXIV:
            documents = load_arxiv_documents(
                data_source.identifier, doc_id, data_source.additional_metadata
            )
        elif data_source.source_type == SourceType.WIKIPEDIA:
            documents = load_wikipedia_documents(
                data_source.identifier,
                doc_id,
                additional_metadata=data_source.additional_metadata,
            )
        elif data_source.source_type == SourceType.WEBSITE:
            documents = load_website_documents(
                data_source.identifier, doc_id, data_source.additional_metadata
            )
        else:
            logger.warning(f"Unknown source type: {data_source.source_type}")
            continue

        # Split and add documents to vectorstore
        split_and_add_documents(vectorstore, documents, chunk_size, chunk_overlap)

        doc_id += 1

    logger.info(f"Finished hydrating vector database with {doc_id - 1} documents")