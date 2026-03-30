"""
config.py
---------
Central configuration loaded from environment variables.
All other modules import from here — never read os.environ directly elsewhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "clinical_notes")

    # LLM
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Embedding
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    # all-MiniLM-L6-v2 produces 384-dimensional vectors
    # BioBERT variant produces 768-dimensional vectors
    # Update VECTOR_SIZE if you switch models
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "384"))

    # Ingestion
    CHUNK_TOKEN_LIMIT: int = int(os.getenv("CHUNK_TOKEN_LIMIT", "256"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))
    DOCUMENT_STORE_PATH: str = os.getenv("DOCUMENT_STORE_PATH", "./data/document_store.json")

    # Dataset
    DATASET_NAME: str = os.getenv("DATASET_NAME", "AGBonnet/augmented-clinical-notes")
    DATASET_SPLIT: str = os.getenv("DATASET_SPLIT", "train")

    APP_TITLE: str = os.getenv("APP_TITLE", "Clinical RAG API")
    APP_VERSION: str = os.getenv("APP_VERSION", "0.1.0")


config = Config()