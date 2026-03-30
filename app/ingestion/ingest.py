"""
ingest.py
---------
Main ingestion pipeline. Orchestrates all Phase 2 steps:

    1. Load dataset from HuggingFace
    2. Build and save the document store (idx → full_note)
    3. For each row:
        a. Parse summary JSON → extract filter metadata
        b. Parse conversation → chunk into turn-pair chunks
        c. Batch embed chunks
        d. Batch upsert to Qdrant with metadata payload
    4. Report final stats

Run this script once before starting the API:
    python -m ingestion.ingest

Optional flags (set via env vars or CLI args):
    RECREATE_COLLECTION=true  → drop and recreate Qdrant collection
    MAX_ROWS=100              → limit rows for testing (0 = all rows)

Progress is logged at INFO level. Set LOG_LEVEL=DEBUG for verbose output.
"""

import logging
import os
import sys
import time
from typing import List

from qdrant_client import QdrantClient
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from ingestion.chunker import Chunk, HybridChunker, TokenCounter
from ingestion.document_store import DocumentStore
from ingestion.indexer import Embedder, QdrantIndexer
from ingestion.summary_parser import parse_summary

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_ingestion(
    max_rows: int = 0,
    recreate_collection: bool = False,
    embedding_batch_size: int = 64,
    upsert_batch_size: int = 100,
) -> None:
    """
    Full ingestion pipeline.

    Parameters
    ----------
    max_rows              : Limit rows processed (0 = all). Use 100 for testing.
    recreate_collection   : Drop and recreate Qdrant collection before indexing.
    embedding_batch_size  : Texts per embedding forward pass.
    upsert_batch_size     : Points per Qdrant upsert call.
    """
    start_time = time.time()

    # ── Step 1: Load dataset ───────────────────────────────────────────────────
    logger.info(f"Loading dataset: {config.DATASET_NAME} (split: {config.DATASET_SPLIT})")
    from datasets import load_dataset

    dataset = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT)

    if max_rows > 0:
        dataset = dataset.select(range(min(max_rows, len(dataset))))
        logger.info(f"Limiting to {len(dataset)} rows for testing.")
    else:
        logger.info(f"Full dataset: {len(dataset)} rows.")

    # ── Step 2: Build document store ──────────────────────────────────────────
    doc_store = DocumentStore(path=config.DOCUMENT_STORE_PATH)
    doc_store.build_from_dataset(dataset)
    doc_store.save()

    # ── Step 3: Initialise components ─────────────────────────────────────────
    logger.info("Initialising embedder...")
    embedder = Embedder(model_name=config.EMBEDDING_MODEL)

    logger.info(f"Detected vector size: {embedder.vector_size}")

    logger.info("Initialising token counter...")
    token_counter = TokenCounter(model_name=config.EMBEDDING_MODEL)

    chunker = HybridChunker(
        token_counter=token_counter,
        token_limit=config.CHUNK_TOKEN_LIMIT,
        overlap_pairs=1,
    )

    logger.info(f"Connecting to Qdrant at {config.QDRANT_URL}...")
    qdrant_client = QdrantClient(url=config.QDRANT_URL)

    indexer = QdrantIndexer(
        client=qdrant_client,
        collection_name=config.QDRANT_COLLECTION_NAME,
        vector_size=embedder.vector_size,
    )
    indexer.create_collection(recreate=recreate_collection)

    # ── Step 4: Main ingestion loop ────────────────────────────────────────────
    # Accumulate batches before embedding + upserting
    pending_chunks: List[Chunk] = []
    pending_metadata: List[dict] = []

    total_chunks = 0
    skipped_rows = 0

    def flush_batch():
        """Embed and upsert the current pending batch."""
        nonlocal total_chunks
        if not pending_chunks:
            return

        texts = [c.text for c in pending_chunks]
        vectors = embedder.embed(texts, batch_size=embedding_batch_size)
        indexer.upsert_batch(pending_chunks, vectors, pending_metadata)

        total_chunks += len(pending_chunks)
        pending_chunks.clear()
        pending_metadata.clear()

    logger.info("Starting ingestion loop...")

    for row in tqdm(dataset, desc="Ingesting rows", unit="row"):
        idx = str(row.get("idx", "")).strip()

        if not idx:
            skipped_rows += 1
            continue

        # ── a. Parse summary metadata (for pre-filters) ───────────────────────
        metadata = parse_summary(row.get("summary", ""))

        # ── b. Chunk conversation ─────────────────────────────────────────────
        conversation = row.get("conversation", "")
        chunks = chunker.chunk(idx=idx, conversation=conversation)

        if not chunks:
            print(f"SKIPPED — no chunks for idx: {idx}")
            print(f"Conversation preview: {conversation[:200]}")
            skipped_rows += 1
            continue

        # ── c. Accumulate for batch processing ────────────────────────────────
        for chunk in chunks:
            pending_chunks.append(chunk)
            pending_metadata.append(metadata)

            # Flush when batch is full
            if len(pending_chunks) >= upsert_batch_size:
                flush_batch()

    # Final flush — process any remaining items
    flush_batch()

    # ── Step 5: Report stats ───────────────────────────────────────────────────
    elapsed = time.time() - start_time
    indexed_count = indexer.count()

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info(f"  Rows processed   : {len(dataset) - skipped_rows}")
    logger.info(f"  Rows skipped     : {skipped_rows}")
    logger.info(f"  Chunks created   : {total_chunks}")
    logger.info(f"  Qdrant points    : {indexed_count}")
    logger.info(f"  Document store   : {len(doc_store)} entries")
    logger.info(f"  Time elapsed     : {elapsed:.1f}s")
    logger.info("=" * 60)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    max_rows = int(os.getenv("MAX_ROWS", "0"))
    recreate = os.getenv("RECREATE_COLLECTION", "false").lower() == "true"

    run_ingestion(
        max_rows=max_rows,
        recreate_collection=recreate,
    )