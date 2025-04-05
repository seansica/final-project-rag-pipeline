"""Singleton pattern to share a single embedding model instance across evaluators."""
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger
import torch

# Global embedding model instance
_embedding_model = None

def get_embeddings():
    """Returns a singleton instance of HuggingFaceEmbeddings model.
    
    This ensures we only load the model once, saving memory and improving performance.
    """
    global _embedding_model
    
    if _embedding_model is None:
        logger.info("Initializing shared embeddings model (one-time)")
        
        # Use CPU if CUDA memory is an issue
        if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] < 1 * 1024 * 1024 * 1024:  # 1GB free memory threshold
            logger.warning("Low CUDA memory detected, using CPU for embeddings")
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        logger.info(f"Using device: {device} for embeddings model")
        
        # Initialize with sensible defaults for memory efficiency
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": 8}  # Lower batch size to reduce memory usage
        )
        
        logger.info("Embedding model initialized successfully")
    
    return _embedding_model

def clear_embeddings():
    """Force clear the embedding model from memory."""
    global _embedding_model
    
    if _embedding_model is not None:
        logger.info("Explicitly clearing embedding model from memory")
        del _embedding_model
        _embedding_model = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")