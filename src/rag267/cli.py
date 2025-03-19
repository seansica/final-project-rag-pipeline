import os
import json
from typing import List, Optional, Union
from pathlib import Path

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from rag267.vector_db_hydration import (
    initialize_vector_db,
    hydrate_vector_db,
    DataSource,
    SourceType,
)
from rag267.rag import RAGSystem

# Create typer app
app = typer.Typer(
    help="RAG system for generating responses for engineering and marketing teams",
    add_completion=False,
)

console = Console()


def get_project_root() -> Path:
    """Get the root directory of the project."""
    # Assumes this file is in src/rag267/cli.py
    return Path(__file__).parent.parent.parent


def load_data_sources_from_config(config_path: Union[str, Path]) -> list[DataSource]:
    """Load data sources from a configuration file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading configuration file: {e}")
        return []

    data_sources = []

    # Add ArXiv papers
    for arxiv_id in config.get("arxiv", []):
        data_sources.append(
            DataSource(identifier=arxiv_id, source_type=SourceType.ARXIV)
        )

    # Add Wikipedia articles
    for query in config.get("wikipedia", []):
        data_sources.append(
            DataSource(identifier=query, source_type=SourceType.WIKIPEDIA)
        )

    # Add websites
    for url in config.get("website", []):
        data_sources.append(DataSource(identifier=url, source_type=SourceType.WEBSITE))

    return data_sources


def create_data_sources(
    config_path: Optional[Path] = None,
    arxiv_ids: Optional[List[str]] = None,
    wikipedia_queries: Optional[List[str]] = None,
    website_urls: Optional[List[str]] = None,
) -> List[DataSource]:
    """Create data sources from config file and/or command line arguments."""
    data_sources = []

    # Load from config file first
    if config_path and config_path.exists():
        config_sources = load_data_sources_from_config(config_path)
        data_sources.extend(config_sources)

    # Add command line sources if provided

    # Add ArXiv papers
    if arxiv_ids:
        for arxiv_id in arxiv_ids:
            data_sources.append(
                DataSource(identifier=arxiv_id, source_type=SourceType.ARXIV)
            )

    # Add Wikipedia articles
    if wikipedia_queries:
        for query in wikipedia_queries:
            data_sources.append(
                DataSource(identifier=query, source_type=SourceType.WIKIPEDIA)
            )

    # Add websites
    if website_urls:
        for url in website_urls:
            data_sources.append(
                DataSource(identifier=url, source_type=SourceType.WEBSITE)
            )

    return data_sources


def get_queries_from_file(query_file: Path) -> List[str]:
    """Get queries from a file."""
    if not query_file.exists():
        print(f"[bold red]Error:[/bold red] Query file {query_file} not found.")
        return []

    queries = []
    try:
        with open(query_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    queries.append(line)
    except Exception as e:
        print(f"[bold red]Error reading query file:[/bold red] {e}")
        return []

    return queries


def interactive_mode(rag_system: RAGSystem):
    """Run the RAG system in interactive mode."""
    console.print(Panel("[bold]RAG System Interactive Mode[/bold]", expand=False))
    console.print("Type [bold green]exit[/bold green] to quit")
    console.print("Type [bold green]reload[/bold green] to reload templates")

    while True:
        query = console.input("\n[bold cyan]Enter your query:[/bold cyan] ")

        if query.lower() == "exit":
            break

        if query.lower() == "reload":
            rag_system.reload_templates()
            console.print("[bold green]Templates reloaded successfully[/bold green]")
            continue

        console.print("\nRetrieving documents and generating responses...")

        try:
            # Get responses
            responses = rag_system.generate_responses(query)

            # Print responses
            console.print(
                Panel(
                    Markdown(responses["engineering"]),
                    title="[bold]Engineering Response[/bold]",
                    expand=False,
                )
            )

            console.print(
                Panel(
                    Markdown(responses["marketing"]),
                    title="[bold]Marketing Response[/bold]",
                    expand=False,
                )
            )

            # Get sources
            console.print("[bold]Sources:[/bold]")
            for source in rag_system.get_document_sources(query):
                console.print(f"- {source}")

        except Exception as e:
            console.print(f"[bold red]Error generating responses:[/bold red] {e}")


@app.command()
def run(
    # Data source arguments
    config: Optional[Path] = typer.Option(
        "data_sources.json",
        help="Path to data sources configuration file",
        exists=False,
    ),
    arxiv: Optional[List[str]] = typer.Option(None, help="ArXiv paper IDs to include"),
    wikipedia: Optional[List[str]] = typer.Option(
        None, help="Wikipedia queries to include"
    ),
    website: Optional[List[str]] = typer.Option(None, help="Website URLs to include"),
    # Template arguments
    engineering_template: Optional[Path] = typer.Option(
        "templates/engineering_template.txt",
        help="Path to engineering template file",
        exists=False,
    ),
    marketing_template: Optional[Path] = typer.Option(
        "templates/marketing_template.txt",
        help="Path to marketing template file",
        exists=False,
    ),
    # Vector DB arguments
    embedding_model: str = typer.Option(
        "multi-qa-mpnet-base-dot-v1", help="Embedding model to use"
    ),
    chunk_size: int = typer.Option(128, help="Size of text chunks"),
    chunk_overlap: int = typer.Option(0, help="Overlap between chunks"),
    top_k: int = typer.Option(4, help="Number of documents to retrieve"),
    # LLM arguments
    mistral_model: str = typer.Option(
        "mistralai/Mistral-7B-Instruct-v0.2", help="Mistral model to use"
    ),
    disable_mistral: bool = typer.Option(False, help="Disable Mistral model"),
    disable_cohere: bool = typer.Option(False, help="Disable Cohere model"),
    # Query arguments
    query: Optional[str] = typer.Option(None, help="Query to run"),
    query_file: Optional[Path] = typer.Option(
        None, help="File containing queries, one per line", exists=False
    ),
    # Mode argument
    interactive: bool = typer.Option(False, help="Run in interactive mode"),
) -> None:
    """
    Run the RAG system with the specified configuration.
    """
    # Convert paths to absolute paths if they're not
    if config and not config.is_absolute():
        config = Path.cwd() / config

    if engineering_template and not engineering_template.is_absolute():
        engineering_template = Path.cwd() / engineering_template

    if marketing_template and not marketing_template.is_absolute():
        marketing_template = Path.cwd() / marketing_template

    if query_file and not query_file.is_absolute():
        query_file = Path.cwd() / query_file

    # Create data sources
    data_sources = create_data_sources(
        config_path=config if config and config.exists() else None,
        arxiv_ids=arxiv,
        wikipedia_queries=wikipedia,
        website_urls=website,
    )

    if not data_sources:
        print(
            "[bold red]Error:[/bold red] No data sources specified. Please provide at least one data source."
        )
        raise typer.Exit(code=1)

    # Ensure templates directory exists
    template_dir = Path.cwd() / "templates"
    template_dir.mkdir(exist_ok=True)

    # Create default template files if they don't exist
    if engineering_template and not engineering_template.exists():
        engineering_template.parent.mkdir(parents=True, exist_ok=True)
        with open(engineering_template, "w") as f:
            f.write(
                """[INST]You are an AI assistant for a technology company, answering questions for the engineering team.
Please provide a detailed and technically precise answer to the question below, based solely on the provided context.

CONTEXT:
{context}

QUESTION:
{question}

Provide a comprehensive answer with technical details, methodologies, and specific information from the context.
Include relevant technical terms and cite specific information from the context when applicable.
Make sure your answer is structured and emphasizes the technical aspects of the topic.
Provide examples or code snippets when appropriate to illustrate concepts.[/INST]"""
            )

    if marketing_template and not marketing_template.exists():
        marketing_template.parent.mkdir(parents=True, exist_ok=True)
        with open(marketing_template, "w") as f:
            f.write(
                """[INST]You are an AI assistant for a technology company, answering questions for the marketing team.
Please provide a clear, concise, and accessible answer to the question below, based solely on the provided context.

CONTEXT:
{context}

QUESTION:
{question}

Provide a straightforward answer that's easy to understand for non-technical audiences.
Avoid overly technical jargon and focus on the practical applications and benefits.
Highlight the value proposition and business impact where relevant.
Use simple analogies or real-world examples to explain complex concepts when necessary.[/INST]"""
            )

    # Initialize vector database
    console.print(
        f"Initializing vector database with embedding model '{embedding_model}'..."
    )
    vectorstore = initialize_vector_db(
        embedding_model_name=embedding_model,
        collection_name="rag_tech_db",
        in_memory=True,
        force_recreate=True,
    )

    # Hydrate vector database
    console.print(f"Hydrating vector database with {len(data_sources)} data sources...")
    hydrate_vector_db(
        vectorstore=vectorstore,
        data_sources=data_sources,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Initialize RAG system
    cohere_api_key = os.environ.get("COHERE_API_KEY", None)
    if not cohere_api_key and not disable_cohere:
        console.print(
            "[bold yellow]Warning:[/bold yellow] COHERE_API_KEY environment variable not set. Disabling Cohere model."
        )
        disable_cohere = True

    console.print("Initializing RAG system...")
    rag_system = RAGSystem(
        vectorstore=vectorstore,
        engineering_template_path=engineering_template,
        marketing_template_path=marketing_template,
        cohere_api_key=cohere_api_key,
        use_mistral=not disable_mistral,
        use_cohere=not disable_cohere,
        mistral_model_name=mistral_model,
        top_k=top_k,
    )

    # Run in interactive mode if specified
    if interactive:
        interactive_mode(rag_system)
        return

    # Get queries
    queries = []
    if query:
        queries.append(query)
    if query_file and query_file.exists():
        file_queries = get_queries_from_file(query_file)
        queries.extend(file_queries)

    if not queries:
        console.print(
            "[bold yellow]Warning:[/bold yellow] No queries specified. Starting interactive mode."
        )
        interactive_mode(rag_system)
        return

    # Run queries
    for i, q in enumerate(queries):
        console.print(Panel(f"[bold]Query {i + 1}:[/bold] {q}", expand=False))

        try:
            # Get responses
            responses = rag_system.generate_responses(q)

            # Print responses
            console.print(
                Panel(
                    Markdown(responses["engineering"]),
                    title="[bold]Engineering Response[/bold]",
                    expand=False,
                )
            )

            console.print(
                Panel(
                    Markdown(responses["marketing"]),
                    title="[bold]Marketing Response[/bold]",
                    expand=False,
                )
            )

            # Get sources
            console.print("[bold]Sources:[/bold]")
            for source in rag_system.get_document_sources(q):
                console.print(f"- {source}")

        except Exception as e:
            console.print(f"[bold red]Error generating responses:[/bold red] {e}")


@app.command()
def create_templates():
    """Create default template files in the templates directory."""
    template_dir = Path.cwd() / "templates"
    template_dir.mkdir(exist_ok=True)

    engineering_template_path = template_dir / "engineering_template.txt"
    marketing_template_path = template_dir / "marketing_template.txt"

    # Create engineering template
    with open(engineering_template_path, "w") as f:
        f.write(
            """[INST]You are an AI assistant for a technology company, answering questions for the engineering team.
Please provide a detailed and technically precise answer to the question below, based solely on the provided context.

CONTEXT:
{context}

QUESTION:
{question}

Provide a comprehensive answer with technical details, methodologies, and specific information from the context.
Include relevant technical terms and cite specific information from the context when applicable.
Make sure your answer is structured and emphasizes the technical aspects of the topic.
Provide examples or code snippets when appropriate to illustrate concepts.[/INST]"""
        )

    # Create marketing template
    with open(marketing_template_path, "w") as f:
        f.write(
            """[INST]You are an AI assistant for a technology company, answering questions for the marketing team.
Please provide a clear, concise, and accessible answer to the question below, based solely on the provided context.

CONTEXT:
{context}

QUESTION:
{question}

Provide a straightforward answer that's easy to understand for non-technical audiences.
Avoid overly technical jargon and focus on the practical applications and benefits.
Highlight the value proposition and business impact where relevant.
Use simple analogies or real-world examples to explain complex concepts when necessary.[/INST]"""
        )

    console.print(
        "[bold green]Templates created successfully in the templates directory.[/bold green]"
    )


@app.command()
def create_config():
    """Create a default data sources configuration file."""
    config_path = Path.cwd() / "data_sources.json"

    config = {
        "arxiv": ["2104.07567", "2105.03011"],
        "wikipedia": [
            "Generative Artificial Intelligence",
            "Large Language Models",
            "Retrieval Augmented Generation",
        ],
        "website": [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2020-10-29-odqa/",
        ],
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    console.print(
        f"[bold green]Default configuration created at {config_path}[/bold green]"
    )
    console.print("Edit this file to add more data sources.")


if __name__ == "__main__":
    app()
