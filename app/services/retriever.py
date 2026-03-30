"""
retriever.py
------------
Retrieves relevant clinical cases for a given patient query.

Flow:
    1. Embed the query using the same model used during ingestion
    2. Optionally apply pre-filters on Qdrant payload fields
    3. Search Qdrant for top-k similar chunks
    4. Apply score threshold — drop weak matches
    5. Deduplicate by idx — multiple chunks from same case → one case
    6. Fetch full_notes from document store
    7. Return clean result dict for the generator

Usage:
    retriever = Retriever()
    result = retriever.retrieve(
        query="severe knee pain when walking",
        filters={"patient_sex": "Male", "patient_age_group": "middle_aged"},
        top_k=5
    )
    # result["notes"]  → list of full_note strings (LLM context)
    # result["cases"]  → list of dicts with idx, score, chunk_text (metadata)
"""

import json
import logging
import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.document_store import DocumentStore

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "clinical_notes")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DOCUMENT_STORE_PATH = os.getenv("DOCUMENT_STORE_PATH", "./data/document_store.json")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))


# ── Retriever ──────────────────────────────────────────────────────────────────

class Retriever:
    """
    Handles all retrieval logic — embedding, filtering, searching, fetching.

    Initialised once at startup and reused across all queries.
    Loading the embedding model is expensive (~2-3s) — do it once only.
    """

    def __init__(self):
        logger.info("Initialising Retriever...")

        # Load embedding model — same model used during ingestion
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self._model = SentenceTransformer(EMBEDDING_MODEL)

        # Connect to Qdrant
        logger.info(f"Connecting to Qdrant at {QDRANT_URL}...")
        self._client = QdrantClient(url=QDRANT_URL)

        # Load document store into memory
        logger.info(f"Loading document store from {DOCUMENT_STORE_PATH}...")
        self._doc_store = DocumentStore(path=DOCUMENT_STORE_PATH)
        self._doc_store.load()

        logger.info("Retriever ready.")

    # ── Public interface ───────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        filters: dict = None,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = SCORE_THRESHOLD,
    ) -> dict:
        """
        Retrieve relevant clinical cases for a patient query.

        Parameters
        ----------
        query            : Plain English patient complaint
        filters          : Optional payload filters e.g.
                           {"patient_sex": "Male", "patient_age_group": "middle_aged"}
                           Only include keys where value is not None.
        top_k            : Number of chunks to retrieve from Qdrant
        score_threshold  : Minimum similarity score to keep (0 to 1)

        Returns
        -------
        {
            "notes": ["full note text 1", ...],   # passed to LLM as context
            "cases": [                             # metadata for inspection
                {
                    "idx": "155216",
                    "score": 0.74,
                    "chunk_text": "Doctor: ... Patient: ...",
                    "patient_sex": "Female",
                    "patient_age_group": "adolescent",
                    "primary_diagnosis": "Tardive dystonia"
                },
                ...
            ]
        }
        Returns {"notes": [], "cases": []} if nothing relevant is found.
        """
        if not query or not query.strip():
            logger.warning("Empty query received — returning empty result.")
            return {"notes": [], "cases": []}

        # ── Step 1: Embed the query ────────────────────────────────────────────
        query_vector = self._embed(query)

        # ── Step 2: Build Qdrant filter ────────────────────────────────────────
        qdrant_filter = self._build_filter(filters)

        # ── Step 3: Search Qdrant ──────────────────────────────────────────────
        raw_results = self._search(query_vector, qdrant_filter, top_k)
        raw_results = self._search(query_vector, qdrant_filter, top_k)

        # ── Step 4: Apply score threshold ─────────────────────────────────────
        filtered_results = [r for r in raw_results if r.score >= score_threshold]

        if not filtered_results:
            logger.info(
                f"No results above score threshold {score_threshold}. "
                f"Best score was {raw_results[0].score:.4f}" if raw_results else "No results at all."
            )
            return {"notes": [], "cases": []}

        # ── Step 5: Deduplicate by idx ─────────────────────────────────────────
        # Multiple chunks from the same case → keep only the best scoring one
        deduplicated = self._deduplicate(filtered_results)

        # ── Step 6: Fetch full notes from document store ───────────────────────
        idx_list = [case["idx"] for case in deduplicated]
        full_notes = self._doc_store.get_many(idx_list)

        # ── Step 7: Build and return result ───────────────────────────────────
        # Attach full_note to each case entry (for reference)
        # Build the notes list in the same order as deduplicated cases
        notes = []
        cases = []

        for case in deduplicated:
            idx = case["idx"]
            note = full_notes.get(idx)

            if not note:
                logger.warning(f"No full note found for idx={idx} — skipping.")
                continue

            notes.append(note)
            cases.append({**case, "full_note_preview": note[:200]})

        logger.info(
            f"Retrieved {len(cases)} unique cases for query: '{query[:60]}...'"
            if len(query) > 60 else f"Retrieved {len(cases)} unique cases for query: '{query}'"
        )

        return {"notes": notes, "cases": cases}

    # ── Private helpers ────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list:
        """Embed a single text string into a vector."""
        return self._model.encode(text).tolist()

    def _build_filter(self, filters: dict) -> Filter | None:
        """
        Build a Qdrant Filter from a plain dict.

        Only fields with non-None, non-empty values are included.
        Returns None if no valid filters — Qdrant will search everything.

        Example input:
            {"patient_sex": "Male", "patient_age_group": None}
        Example output:
            Filter(must=[FieldCondition(key="patient_sex", match=MatchValue(value="Male"))])
        """
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            if value is None or value == "":
                continue
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )

        if not conditions:
            return None

        return Filter(must=conditions)

    def _search(self, query_vector: list, qdrant_filter: Filter | None, top_k: int) -> list:
        """Run the Qdrant vector search and return raw results."""
        results = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        return results.points

    def _deduplicate(self, results: list) -> list[dict]:
        """
        Deduplicate results by idx, keeping the highest scoring chunk per case.

        Returns a list of dicts with idx, score, chunk_text, and metadata.
        Ordered by score descending.
        """
        # Group by idx — keep only the best score per case
        best_per_case = {}

        for result in results:
            payload = result.payload
            idx = payload.get("idx")

            if not idx:
                continue

            if idx not in best_per_case or result.score > best_per_case[idx]["score"]:
                best_per_case[idx] = {
                    "idx": idx,
                    "score": round(result.score, 4),
                    "chunk_text": payload.get("chunk_text", ""),
                    "patient_sex": payload.get("patient_sex"),
                    "patient_age_group": payload.get("patient_age_group"),
                    "patient_age": payload.get("patient_age"),
                    "primary_diagnosis": payload.get("primary_diagnosis"),
                    "visit_motivation": payload.get("visit_motivation"),
                }

        # Sort by score descending
        return sorted(best_per_case.values(), key=lambda x: x["score"], reverse=True)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 60)
    print("RETRIEVER — STANDALONE TEST")
    print("=" * 60)

    retriever = Retriever()

    test_cases = [
        {
            "query": "I have been having severe knee pain for two weeks, worse when I walk",
            "filters": {},
        },
        {
            "query": "16 year old cannot walk, severe weakness in both legs",
            "filters": {"patient_age_group": "adolescent"},
        },
        {
            "query": "neck pain and restricted movement, cannot stand properly",
            "filters": {"patient_sex": "Female"},
        },
    ]

    for i, test in enumerate(test_cases, start=1):
        print(f"\n{'─' * 60}")
        print(f"Test {i}: {test['query']}")
        print(f"Filters: {test['filters']}")
        print(f"{'─' * 60}")

        result = retriever.retrieve(
            query=test["query"],
            filters=test["filters"],
            top_k=5,
        )

        if not result["cases"]:
            print("  No results above threshold.")
            continue

        print(f"\n  {len(result['cases'])} case(s) retrieved:\n")
        for case in result["cases"]:
            print(f"  idx: {case['idx']} | score: {case['score']}")
            print(f"  age_group: {case.get('patient_age_group')} | sex: {case.get('patient_sex')}")
            print(f"  diagnosis: {case.get('primary_diagnosis')}")
            print(f"  chunk preview: {case['chunk_text'][:150]}")
            print(f"  note preview:  {case['full_note_preview']}")
            print()

    print("=" * 60)
    print("Retriever test complete.")
    print("=" * 60)