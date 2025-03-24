from typing import List

from .utils import initialize_vector_db, hydrate_vector_db, DataSource


class VectorDatabaseManager:
    def __init__(
        self,
        embedding_model_name: str = "multi-qa-mpnet-base-dot-v1",
        collection_name: str = "myrag",
        chunk_size: int = 128,
        chunk_overlap: int = 0,
        in_memory: bool = True,
        force_recreate: bool = True,
    ):
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.in_memory = in_memory
        self.force_recreate = force_recreate
        
        # *** VECTORSTORE ***
        self.vectorstore = initialize_vector_db(
            embedding_model_name=embedding_model_name,
            collection_name=collection_name,
            in_memory=in_memory,
            force_recreate=force_recreate,
        )

    def hydrate(self, data_sources: List[DataSource]) -> None:
        hydrate_vector_db(
            self.vectorstore,
            data_sources,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def get_config(self) -> dict:
        return {
            'embedding_model': self.embedding_model_name,
            'collection_name': self.collection_name,
            'in_memory': self.in_memory,
            'force_recreate': self.force_recreate,
            'chunking_strategy': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'splitter_type': "RecursiveCharacterTextSplitter"
            }
        }
