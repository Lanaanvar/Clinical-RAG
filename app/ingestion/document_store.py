"""
document_store.py
-----------------
Builds and manages the separate document store: { idx → full_note }.

Responsibilities:
- Build the store from the HuggingFace dataset
- Persist to disk as JSON (so ingestion doesn't need to re-run)
- Provide fast O(1) lookups at retrieval time

Design decisions:
- Stored as a flat JSON file for simplicity and portability
- idx keys are always stored as strings (safe across JSON serialisation)
- The store is loaded fully into memory at API startup — at 30k rows with
  avg ~5KB per full_note this is ~150MB, acceptable for a single-node setup
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DocumentStore:
    """
    Manages the idx → full_note mapping.

    Usage
    -----
    Building (during ingestion):
        store = DocumentStore(path="./data/document_store.json")
        store.build_from_dataset(dataset)
        store.save()

    Loading (at API startup):
        store = DocumentStore(path="./data/document_store.json")
        store.load()
        full_note = store.get("155216")
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self._store: dict[str, str] = {}

    # ── Build ──────────────────────────────────────────────────────────────────

    def build_from_dataset(self, dataset) -> None:
        """
        Iterate through the HuggingFace dataset and populate the store.

        Parameters
        ----------
        dataset : HuggingFace Dataset object (already loaded and split)
        """
        logger.info("Building document store from dataset...")
        skipped = 0

        for row in dataset:
            idx = str(row.get("idx", "")).strip()
            full_note = row.get("full_note", "")

            if not idx:
                skipped += 1
                continue

            if not full_note or not isinstance(full_note, str):
                skipped += 1
                logger.debug(f"[{idx}] Missing or invalid full_note — skipping.")
                continue

            # Store as-is — cleaning happens at generation time if needed
            self._store[idx] = full_note.strip()

        logger.info(
            f"Document store built: {len(self._store)} entries, {skipped} skipped."
        )

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the store to disk as a JSON file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False)
        size_mb = self.path.stat().st_size / (1024 * 1024)
        logger.info(f"Document store saved to {self.path} ({size_mb:.1f} MB).")

    def load(self) -> None:
        """Load the store from disk into memory."""
        if not self.path.exists():
            raise FileNotFoundError(
                f"Document store not found at {self.path}. "
                "Run the ingestion pipeline first."
            )
        with open(self.path, "r", encoding="utf-8") as f:
            self._store = json.load(f)
        logger.info(
            f"Document store loaded from {self.path}: {len(self._store)} entries."
        )

    # ── Lookup ─────────────────────────────────────────────────────────────────

    def get(self, idx: str) -> Optional[str]:
        """
        Retrieve the full_note for a given idx.

        Parameters
        ----------
        idx : str — must match exactly the key stored during build

        Returns None if not found (do not raise — let the caller decide).
        """
        return self._store.get(str(idx))

    def get_many(self, idx_list: list[str]) -> dict[str, str]:
        """
        Retrieve full_notes for a list of idx values.
        Returns only the ones found — missing keys are silently skipped.
        """
        result = {}
        for idx in idx_list:
            note = self.get(idx)
            if note:
                result[str(idx)] = note
            else:
                logger.warning(f"Document store miss for idx={idx}")
        return result

    # ── Utilities ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, idx: str) -> bool:
        return str(idx) in self._store

    @property
    def is_loaded(self) -> bool:
        return len(self._store) > 0