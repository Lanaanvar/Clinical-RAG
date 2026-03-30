"""
rag_pipeline.py
---------------
Orchestrates the full RAG pipeline — retrieval followed by generation.

This is the single entry point the FastAPI route will call in Phase 5.

Usage:
    pipeline = RAGPipeline()
    result = pipeline.run(
        query="I have severe knee pain when walking",
        filters={"patient_sex": "Male"},
        top_k=5
    )
    print(result["answer"])
    print(result["cases_used"])
"""

import logging
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.retriever import Retriever
from services.generator import Generator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Wires the Retriever and Generator together into a single callable pipeline.

    Both components are initialised once and reused across all queries.
    This matters for the API — you don't want to reload the embedding
    model on every request.
    """

    def __init__(self):
        logger.info("Initialising RAG Pipeline...")
        self._retriever = Retriever()
        self._generator = Generator()
        logger.info("RAG Pipeline ready.")

    def run(
        self,
        query: str,
        filters: dict = None,
        top_k: int = 5,
    ) -> dict:
        """
        Run the full RAG pipeline for a patient query.

        Parameters
        ----------
        query   : Plain English patient complaint
        filters : Optional pre-filters e.g. {"patient_sex": "Male"}
        top_k   : Number of chunks to retrieve before deduplication

        Returns
        -------
        {
            "answer"        : str   — grounded LLM response
            "cases_used"    : int   — number of cases passed to LLM
            "cases"         : list  — metadata for each retrieved case
            "query"         : str   — original query
            "model"         : str   — LLM model used
            "retrieval_ms"  : int   — time taken for retrieval in ms
            "generation_ms" : int   — time taken for generation in ms
            "total_ms"      : int   — total pipeline time in ms
        }
        """
        total_start = time.time()

        # ── Step 1: Retrieve ───────────────────────────────────────────────────
        retrieval_start = time.time()
        retrieval_result = self._retriever.retrieve(
            query=query,
            filters=filters or {},
            top_k=top_k,
        )
        retrieval_ms = int((time.time() - retrieval_start) * 1000)

        notes = retrieval_result["notes"]
        cases = retrieval_result["cases"]

        logger.info(
            f"Retrieval completed in {retrieval_ms}ms — "
            f"{len(cases)} case(s) found."
        )

        # ── Step 2: Generate ───────────────────────────────────────────────────
        generation_start = time.time()
        generation_result = self._generator.generate(
            query=query,
            notes=notes,
        )
        generation_ms = int((time.time() - generation_start) * 1000)

        total_ms = int((time.time() - total_start) * 1000)

        logger.info(
            f"Generation completed in {generation_ms}ms. "
            f"Total pipeline: {total_ms}ms."
        )

        # ── Step 3: Return combined result ─────────────────────────────────────
        return {
            "answer":        generation_result["answer"],
            "cases_used":    generation_result["cases_used"],
            "cases":         cases,
            "query":         query,
            "model":         generation_result["model"],
            "retrieval_ms":  retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms":      total_ms,
        }


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 60)
    print("RAG PIPELINE — END TO END TEST")
    print("=" * 60)

    pipeline = RAGPipeline()

    test_cases = [
        {
            "label": "Knee pain — no filters",
            "query": "I have been having severe knee pain for two weeks, it gets worse when I walk and I cannot put weight on it",
            "filters": {},
        },
        {
            "label": "Leg weakness — adolescent filter",
            "query": "teenager who cannot walk, severe weakness and pain in both legs",
            "filters": {"patient_age_group": "adolescent"},
        },
        {
            "label": "Hip pain — female filter",
            "query": "severe hip pain and restricted movement, difficulty walking",
            "filters": {"patient_sex": "Female"},
        },
    ]

    for test in test_cases:
        print(f"\n{'─' * 60}")
        print(f"Test: {test['label']}")
        print(f"Query: {test['query']}")
        print(f"Filters: {test['filters']}")
        print(f"{'─' * 60}")

        result = pipeline.run(
            query=test["query"],
            filters=test["filters"],
            top_k=5,
        )

        print(f"\nCases retrieved : {result['cases_used']}")
        print(f"Retrieval time  : {result['retrieval_ms']}ms")
        print(f"Generation time : {result['generation_ms']}ms")
        print(f"Total time      : {result['total_ms']}ms")
        print(f"\nCases used (idx): {[c['idx'] for c in result['cases']]}")

        print(f"\nAnswer:\n")
        print(result["answer"])
        print()

    print("=" * 60)
    print("End to end test complete.")
    print("=" * 60)