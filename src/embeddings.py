"""
DocuMind - Embeddings Module
============================
Initializes and manages HuggingFace Sentence-Transformer embeddings.
Uses the lightweight, high-performance 'all-MiniLM-L6-v2' model which runs
entirely locally on CPU with zero API costs and strong semantic search accuracy.
"""

import os
import logging
from typing import Optional
from functools import lru_cache

# Suppress Windows symlink warnings and enable offline caching if local snapshot exists
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# If local snapshot of model exists in HuggingFace cache, prefer offline mode to prevent network timeouts
cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2")
if os.path.exists(cache_dir):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

import config

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages loading and instantiation of vector embedding models.
    Implements CPU optimization and singleton caching to prevent redundant reloading.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding manager.

        Args:
            model_name (str, optional): HuggingFace model repo name.
                                        Defaults to config.DEFAULT_EMBEDDING_MODEL ('all-MiniLM-L6-v2').
        """
        self.model_name = model_name or config.DEFAULT_EMBEDDING_MODEL
        self._embeddings = None

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Retrieve or initialize the HuggingFaceEmbeddings instance.
        Configures model to run on CPU with normalized embeddings for cosine similarity.

        Returns:
            HuggingFaceEmbeddings: Configured LangChain embedding model.
        """
        if self._embeddings is None:
            logger.info(f"Loading HuggingFace embedding model: '{self.model_name}' on CPU...")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("Embedding model loaded successfully.")

        return self._embeddings


# Global cached helper for convenient access across the app
@lru_cache(maxsize=2)
def get_embedding_model(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """
    Cached factory function for obtaining the HuggingFace embedding model instance.
    Utilizes LRU caching to avoid re-allocating memory across multiple Streamlit reruns.

    Args:
        model_name (str, optional): Name of the embedding model.

    Returns:
        HuggingFaceEmbeddings: Cached embedding instance.
    """
    manager = EmbeddingManager(model_name=model_name)
    return manager.get_embeddings()
