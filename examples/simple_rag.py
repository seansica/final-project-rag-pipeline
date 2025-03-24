"""Simple example of using the RAG267 package."""

from dotenv import load_dotenv

from rag267.indexing.embedding_model import get_embedding_model
from rag267.indexing.split import split_documents
from rag267.rag import create_vectorstore, get_retriever
from rag267.rag.llm import get_mistral_llm, get_cohere_llm
from rag267.indexing.web_scraper import load_web_documents, add_metadata_to_documents
from rag267.rag.rag import RAGPipeline


def main():
    """Run a simple RAG example."""
    # Load environment variables
    load_dotenv()

    # Initialize embedding model
    print("Loading embedding model...")
    embedding_model = get_embedding_model("multi-qa-mpnet-base-dot-v1")

    # Load documents
    print("Loading documents...")
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    ]
    documents = load_web_documents(urls)
    documents, _ = add_metadata_to_documents(documents, "WWW", 1)

    # Split documents
    print("Splitting documents...")
    splits = split_documents(documents, chunk_size=128, chunk_overlap=0)
    print(f"Created {len(splits)} document chunks")

    # Create vector store and retriever
    print("Creating vector store...")
    vectorstore = create_vectorstore(splits, embedding_model)
    retriever = get_retriever(vectorstore)

    # Initialize LLMs
    print("Loading language models...")
    mistral_llm = get_mistral_llm()
    cohere_llm = get_cohere_llm()

    # Create RAG pipeline
    rag_pipeline = RAGPipeline(retriever, mistral_llm, cohere_llm)

    # Ask a question
    question = "What is Chain of Thought?"

    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    # Get engineering answer
    print("\nENGINEERING ANSWER:")
    engineering_answer = rag_pipeline.answer_question(question, "engineering")
    print(engineering_answer)

    # Get marketing answer
    print("\nMARKETING ANSWER:")
    marketing_answer = rag_pipeline.answer_question(question, "marketing")
    print(marketing_answer)

    # Show retrieved context
    print("\nRETRIEVED CONTEXT:")
    retrieved_docs = rag_pipeline.get_retrieved_context(question)
    for i, doc in enumerate(retrieved_docs):
        print(f"\nDocument {i + 1}:")
        print(f"Source: {doc.metadata.get('doc_source', 'Unknown')}")
        print(f"Content: {doc.page_content[:100]}...")


if __name__ == "__main__":
    main()
