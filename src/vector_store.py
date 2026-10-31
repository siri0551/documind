"""
DocuMind - Vector Store Module
==============================
Handles vector index creation, persistence, loading, and semantic similarity search
using FAISS (Facebook AI Similarity Search) and LangChain.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Union
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Custom exception raised when vector store operations encounter errors."""
    pass


class VectorStoreManager:
    """
    Manages FAISS vector store operations: index creation, similarity search,
    disk persistence, and index reloading.
    """

    @staticmethod
    def create_vector_store(
        documents: List[Document],
        embeddings: Embeddings
    ) -> FAISS:
        """
        Creates an in-memory FAISS vector store from a list of Document chunks.

        Args:
            documents (List[Document]): Chunked documents to vectorize.
            embeddings (Embeddings): Embedding model instance.

        Returns:
            FAISS: Initialized FAISS vector store.

        Raises:
            VectorStoreError: If documents list is empty or embedding fails.
        """
        if not documents:
            raise VectorStoreError("Cannot create vector store: Document list is empty.")

        try:
            logger.info(f"Generating embeddings and building FAISS index for {len(documents)} chunks...")
            vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
            logger.info("FAISS vector index built successfully.")
            return vector_store
        except Exception as e:
            logger.error(f"Error creating FAISS vector store: {e}")
            raise VectorStoreError(f"Failed to build vector store: {str(e)}")

    @staticmethod
    def add_documents(
        vector_store: FAISS,
        documents: List[Document]
    ) -> FAISS:
        """
        Appends new document chunks into an existing FAISS vector store.

        Args:
            vector_store (FAISS): Existing FAISS instance.
            documents (List[Document]): New chunks to embed and add.

        Returns:
            FAISS: Updated vector store.
        """
        if not documents:
            return vector_store

        try:
            vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} new chunks to existing vector store.")
            return vector_store
        except Exception as e:
            raise VectorStoreError(f"Failed to add documents to vector store: {str(e)}")

    @staticmethod
    def save_index(
        vector_store: FAISS,
        folder_path: Union[str, Path]
    ) -> str:
        """
        Persists FAISS vector store to disk.

        Args:
            vector_store (FAISS): FAISS instance to save.
            folder_path (Union[str, Path]): Destination directory.

        Returns:
            str: Directory path where index was saved.
        """
        path = Path(folder_path)
        path.mkdir(parents=True, exist_ok=True)
        try:
            vector_store.save_local(str(path))
            logger.info(f"FAISS index successfully saved to {path}")
            return str(path)
        except Exception as e:
            raise VectorStoreError(f"Failed to save FAISS index: {str(e)}")

    @staticmethod
    def load_index(
        folder_path: Union[str, Path],
        embeddings: Embeddings
    ) -> Optional[FAISS]:
        """
        Loads a persisted FAISS vector index from disk.

        Args:
            folder_path (Union[str, Path]): Directory containing index.faiss and index.pkl.
            embeddings (Embeddings): Embedding model matching the saved index.

        Returns:
            Optional[FAISS]: Loaded vector store or None if path does not exist.
        """
        path = Path(folder_path)
        if not path.exists():
            return None

        try:
            vector_store = FAISS.load_local(
                folder_path=str(path),
                embeddings=embeddings,
                allow_dangerous_deserialization=True  # Safe for local application indices
            )
            logger.info(f"FAISS index successfully loaded from {path}")
            return vector_store
        except Exception as e:
            raise VectorStoreError(f"Failed to load FAISS index from {path}: {str(e)}")

    @staticmethod
    def get_retriever(
        vector_store: FAISS,
        k: int = 4,
        score_threshold: Optional[float] = None
    ):
        """
        Returns a configured LangChain Retriever from the FAISS vector store.

        Args:
            vector_store (FAISS): Vector store instance.
            k (int): Number of top documents to retrieve.
            score_threshold (float, optional): Optional similarity score cutoff.

        Returns:
            VectorStoreRetriever: Configured retriever object.
        """
        if score_threshold is not None:
            return vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold}
            )
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
