from .vectordb.utils import DataSource, SourceType

# Define data sources
data_sources = [
    # ArXiv papers
    DataSource(identifier="2005.11401", source_type=SourceType.ARXIV),
    DataSource(identifier="2104.07567", source_type=SourceType.ARXIV),
    # Wikipedia articles
    DataSource(
        identifier="Generative Artificial Intelligence",
        source_type=SourceType.WIKIPEDIA,
    ),
    DataSource(
        identifier="Large Language Models",
        source_type=SourceType.WIKIPEDIA,
        additional_metadata={"category": "AI Models"},
    ),
    # Websites
    DataSource(
        identifier="https://lilianweng.github.io/posts/2023-06-23-agent/",
        source_type=SourceType.WEBSITE,
    ),
    DataSource(
        identifier="https://lilianweng.github.io/posts/2020-10-29-odqa/",
        source_type=SourceType.WEBSITE,
        additional_metadata={"author": "Lilian Weng"},
    ),
]