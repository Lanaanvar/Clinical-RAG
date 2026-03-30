# Quick check in a notebook or throwaway script
from datasets import load_dataset

from config import config

ds = load_dataset(config.DATASET_NAME, split="train")

missing_idx = [i for i, r in enumerate(ds) if not str(r.get("idx", "")).strip()]
missing_note = [i for i, r in enumerate(ds) if not r.get("full_note", "")]

print(f"Missing idx: {len(missing_idx)}")
print(f"Missing full_note: {len(missing_note)}")