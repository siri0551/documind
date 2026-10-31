"""
DocuMind - Configuration Module
===============================
Manages global configuration, model names, default chunking parameters,
and environment variable loading for the DocuMind RAG application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file if present
load_dotenv(dotenv_path=BASE_DIR / ".env")

# ==============================================================================
# Application Metadata
# ==============================================================================
APP_TITLE = "DocuMind – AI PDF Assistant"
APP_TAGLINE = "Interact with your PDF documents with intelligent Retrieval-Augmented Generation"
APP_ICON = "📄"

# ==============================================================================
# Model Configurations
# ==============================================================================
# Supported Gemini Models
AVAILABLE_LLM_MODELS = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# HuggingFace Embeddings
# 'all-MiniLM-L6-v2' is lightweight, produces 384-dim dense vectors,
# runs completely on CPU without API keys, and has top-tier RAG benchmarks.
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ==============================================================================
# Text Splitting & Chunking Parameters
# ==============================================================================
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Number of top relevant document chunks to retrieve per question
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))

# ==============================================================================
# Storage & Paths
# ==============================================================================
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"


def get_google_api_key() -> str:
    """
    Retrieve Google Gemini API Key from environment variables.
    Returns empty string if not found.
    """
    return os.getenv("GOOGLE_API_KEY", "").strip()
