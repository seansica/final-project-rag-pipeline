# RAG267

A Retrieval Augmented Generation (RAG) system built with LangChain and Qdrant.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag267.git
cd rag267

# Create a virtual environment
uv init --lib rag267
source .venv/bin/activate
```

## Environment Setup

Create a `.env` file in the root directory with your API keys:

```
COHERE_API_KEY=your_cohere_api_key
```

## Usage

### Indexing Documents

To index documents for retrieval:

```bash
python -m rag267.cli index --web "https://example.com/page1" "https://example.com/page2" --wiki "Large Language Models" "Retrieval Augmented Generation" --chunk-size 128 --chunk-overlap 0
```

### Querying the System

To query the system:

```bash
python -m rag267.cli query "What is Chain of Thought?" --audience engineering
```

Or for marketing-focused responses:

```bash
python -m rag267.cli query "What is Chain of Thought?" --audience marketing
```

## Package Structure

- `embeddings.py`: Embedding models for vector representation of text
- `chunking.py`: Text chunking utilities for document processing
- `vectorstore.py`: Vector store for document storage and retrieval
- `llm.py`: Language model configurations
- `loaders.py`: Document loaders for various data sources
- `prompts.py`: Prompt templates for RAG system
- `rag.py`: RAG pipeline implementation
- `evaluation.py`: Evaluation metrics for RAG system
- `cli.py`: Command-line interface

## Example

A simple example script is provided in `examples/simple_rag.py`:

```bash
python -m rag267.examples.simple_rag
```

## Dependencies

- langchain
- transformers
- qdrant-client
- cohere
- and more (see pyproject.toml)
