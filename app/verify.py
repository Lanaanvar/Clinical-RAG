"""
verify.py
---------
Phase 3 verification script.

Proves that the ingestion pipeline worked correctly by running
3 test queries against Qdrant and checking the results manually.

Run from the app/ directory:
    python verify.py

What to look for:
- Scores above 0.5 for clearly relevant results
- chunk_text that is semantically related to the query
- full_note that confirms the chunk came from a relevant case
- Deduplication working (fewer unique cases than chunks returned)
"""

import json
import os
import sys
from typing import List

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Config (read directly — no need to import full config module) ──────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "clinical_notes")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DOCUMENT_STORE_PATH = os.getenv("DOCUMENT_STORE_PATH", "./data/document_store.json")
TOP_K = 5

# ── Test queries ───────────────────────────────────────────────────────────────
# Three queries across different clinical domains
TEST_QUERIES = [
    "I have been having severe knee pain for two weeks, it gets worse when I walk",
    "I cannot walk at all, severe weakness in both my legs for the past few months",
    "16 year old with neck pain and restricted movement, cannot stand properly",
]

SEPARATOR = "=" * 70


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_document_store(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[ERROR] Document store not found at: {path}")
        print("Make sure you have run ingest.py before running verify.py")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        store = json.load(f)
    print(f"[OK] Document store loaded — {len(store)} entries\n")
    return store


def load_embedding_model(model_name: str) -> SentenceTransformer:
    print(f"[OK] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[OK] Model loaded — vector size: {model.get_sentence_embedding_dimension()}\n")
    return model


def connect_qdrant(url: str, collection_name: str) -> QdrantClient:
    client = QdrantClient(url=url)
    # Verify the collection exists
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        print(f"[ERROR] Collection '{collection_name}' not found in Qdrant.")
        print(f"Available collections: {collections}")
        print("Make sure you have run ingest.py before running verify.py")
        sys.exit(1)
    count = client.count(collection_name=collection_name).count
    print(f"[OK] Connected to Qdrant — collection '{collection_name}' has {count} points\n")
    return client


def search(
    client: QdrantClient,
    model: SentenceTransformer,
    query: str,
    top_k: int,
) -> list:
    query_vector = model.encode(query).tolist()

    # Newer qdrant-client API
    if hasattr(client, "query_points"):
        resp = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return resp.points if hasattr(resp, "points") else resp

    # Backward compatibility with older qdrant-client API
    return client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )


def deduplicate_by_idx(results: list) -> List[str]:
    """
    Extract unique idx values from results, preserving order.
    Multiple chunks from the same case → one idx entry.
    """
    seen = set()
    unique = []
    for r in results:
        idx = r.payload.get("idx")
        if idx and idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique


def run_verification():
    print(SEPARATOR)
    print("PHASE 3 — VERIFICATION")
    print(SEPARATOR)
    print()

    # ── Setup ──────────────────────────────────────────────────────────────────
    doc_store = load_document_store(DOCUMENT_STORE_PATH)
    model = load_embedding_model(EMBEDDING_MODEL)
    client = connect_qdrant(QDRANT_URL, COLLECTION_NAME)

    # ── Run each test query ────────────────────────────────────────────────────
    for query_num, query in enumerate(TEST_QUERIES, start=1):
        print(SEPARATOR)
        print(f"QUERY {query_num}: {query}")
        print(SEPARATOR)

        results = search(client, model, query, top_k=TOP_K)

        if not results:
            print("[WARNING] No results returned — check your Qdrant index\n")
            continue

        # ── Print each chunk result ────────────────────────────────────────────
        print(f"\nTop {len(results)} chunks returned:\n")
        for i, result in enumerate(results, start=1):
            score = result.score
            payload = result.payload
            idx = payload.get("idx", "N/A")
            chunk_id = payload.get("chunk_id", "N/A")
            chunk_text = payload.get("chunk_text", "")

            # Metadata filters stored in payload
            age_group = payload.get("patient_age_group", "unknown")
            sex = payload.get("patient_sex", "unknown")
            diagnosis = payload.get("primary_diagnosis", "unknown")

            print(f"  [{i}] Score: {score:.4f} | idx: {idx} | chunk: {chunk_id}")
            print(f"       Age group: {age_group} | Sex: {sex} | Diagnosis: {diagnosis}")
            print(f"       Chunk text preview:")
            # Print first 300 chars of chunk text, indented
            preview = chunk_text[:300].replace("\n", "\n       ")
            print(f"       {preview}")
            print()

        # ── Deduplication check ────────────────────────────────────────────────
        unique_idx = deduplicate_by_idx(results)
        print(f"  Deduplication: {len(results)} chunks → {len(unique_idx)} unique cases")
        print(f"  Unique idx values: {unique_idx}\n")

        # ── Fetch full notes from document store ───────────────────────────────
        print(f"  Full notes from document store:\n")
        for idx in unique_idx:
            full_note = doc_store.get(idx)
            if not full_note:
                print(f"  [WARNING] idx {idx} not found in document store")
                continue
            print(f"  --- Case idx: {idx} ---")
            print(f"  {full_note[:400]}")
            print()

        print()

    # ── Summary checklist ──────────────────────────────────────────────────────
    print(SEPARATOR)
    print("VERIFICATION CHECKLIST — review manually:")
    print(SEPARATOR)
    print("  [ ] Scores above 0.5 for clearly relevant results?")
    print("  [ ] chunk_text content is semantically related to each query?")
    print("  [ ] full_note confirms the chunk came from a relevant case?")
    print("  [ ] Deduplication reduced chunks to fewer unique cases?")
    print("  [ ] No errors loading document store or connecting to Qdrant?")
    print()
    print("If all boxes check out → Phase 3 complete, move to Phase 4.")
    print("If results look irrelevant → check EMBEDDING_MODEL matches ingestion.")
    print(SEPARATOR)


if __name__ == "__main__":
    run_verification()