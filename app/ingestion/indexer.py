"""
indexer.py
----------
Embeds chunks and upserts them into Qdrant with their metadata payload.

Each Qdrant point contains:
    vector  : embedding of the chunk text (small, for retrieval)
    payload : {
        idx               : str   ← links back to document store
        chunk_id          : str   ← unique per chunk
        chunk_text        : str   ← stored for inspection/debugging
        patient_age       : int   ← from summary (pre-filter)
        patient_age_group : str   ← from summary (pre-filter)
        patient_sex       : str   ← from summary (pre-filter)
        visit_motivation  : str   ← from summary (pre-filter)
        primary_diagnosis : str   ← from summary (pre-filter)
    }

Batching:
- Embeddings are computed in batches (default: 64 texts at once)
- Qdrant upserts are also batched (default: 100 points at once)
- These two batch sizes can differ — embedding batches are GPU/CPU bound,
  Qdrant batches are network/IO bound
"""

import logging
import uuid
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


# ── Embedder ───────────────────────────────────────────────────────────────────

class Embedder:
    """
    Wraps a sentence-transformers model for batch text embedding.

    Parameters
    ----------
    model_name : HuggingFace model name or local path
    """

    def __init__(self, model_name: str):
        logger.info(f"Loading embedding model: {model_name}")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    def embed(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Embed a list of texts. Returns a list of float vectors.

        Parameters
        ----------
        texts      : List of strings to embed
        batch_size : Number of texts to encode per forward pass

        Note: SentenceTransformer.encode() handles batching internally
        when you pass batch_size, but we call it once with all texts
        to let the library optimise GPU usage.
        """
        if not texts:
            return []

        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # Convert numpy arrays to plain Python lists for Qdrant
        return [v.tolist() for v in vectors]

    @property
    def vector_size(self) -> int:
        return self._model.get_sentence_embedding_dimension()


# ── Qdrant collection manager ──────────────────────────────────────────────────

class QdrantIndexer:
    """
    Manages Qdrant collection creation and point upserts.

    Parameters
    ----------
    client          : QdrantClient instance
    collection_name : Name of the Qdrant collection to use
    vector_size     : Dimension of embedding vectors
    """

    def __init__(self, client: QdrantClient, collection_name: str, vector_size: int):
        self._client = client
        self._collection_name = collection_name
        self._vector_size = vector_size

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the Qdrant collection if it doesn't exist.

        Parameters
        ----------
        recreate : If True, delete and recreate the collection.
                   Use this during development to re-index cleanly.
                   Set to False in production to avoid data loss.
        """
        existing = [c.name for c in self._client.get_collections().collections]

        if self._collection_name in existing:
            if recreate:
                logger.warning(
                    f"Recreating collection '{self._collection_name}' — all existing data will be lost."
                )
                self._client.delete_collection(self._collection_name)
            else:
                logger.info(
                    f"Collection '{self._collection_name}' already exists. Upserting into existing collection."
                )
                return

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=self._vector_size,
                distance=Distance.COSINE,
                # COSINE distance is standard for sentence embeddings
                # It measures the angle between vectors, not magnitude
                # This is appropriate for semantic similarity tasks
            ),
        )
        logger.info(f"Collection '{self._collection_name}' created (vector size: {self._vector_size}).")

    def upsert_batch(self, chunks: List[Chunk], vectors: List[List[float]], metadata_list: List[dict]) -> None:
        """
        Upsert a batch of chunks into Qdrant.

        Parameters
        ----------
        chunks        : List of Chunk objects
        vectors       : Corresponding embedding vectors (same order as chunks)
        metadata_list : Corresponding metadata dicts (filter fields from summary)
        """
        if not chunks:
            return

        points = []
        for chunk, vector, metadata in zip(chunks, vectors, metadata_list):
            # Build the full payload
            payload = {
                # Core linking fields
                "idx": chunk.idx,
                "chunk_id": chunk.chunk_id,
                "chunk_text": chunk.text,   # stored for debugging — not used in retrieval

                # Pre-filter fields from summary
                "patient_age": metadata.get("patient_age"),
                "patient_age_group": metadata.get("patient_age_group"),
                "patient_sex": metadata.get("patient_sex"),
                "visit_motivation": metadata.get("visit_motivation"),
                "primary_diagnosis": metadata.get("primary_diagnosis"),
            }

            # Remove None values — Qdrant handles missing fields gracefully
            # but explicit Nones can cause issues with some filter operations
            payload = {k: v for k, v in payload.items() if v is not None}

            points.append(PointStruct(
                # Use chunk_id as the Qdrant point ID (must be str or int)
                # We use a UUID derived from chunk_id for guaranteed uniqueness
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                vector=vector,
                payload=payload,
            ))

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,  # Wait for write to complete before returning
        )

    def count(self) -> int:
        """Return the number of indexed points in the collection."""
        result = self._client.count(collection_name=self._collection_name)
        return result.count