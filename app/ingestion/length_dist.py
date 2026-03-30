import pandas as pd
from datasets import load_dataset
import tiktoken
import numpy as np

# 1. Load the dataset
print("Loading dataset...")
dataset = load_dataset("AGBonnet/augmented-clinical-notes", split="train")
df = pd.DataFrame(dataset)

# 2. Initialize the tokenizer (e.g., cl100k_base for OpenAI's text-embedding-3)
encoding = tiktoken.get_encoding("cl100k_base")

def get_token_count(text):
    if not text or not isinstance(text, str):
        return 0
    return len(encoding.encode(text))

# 3. Calculate lengths
print("Calculating tokens (this may take a minute for 30k rows)...")
df['full_note_len'] = df['full_note'].apply(get_token_count)
df['conversation_len'] = df['conversation'].apply(get_token_count)

# 4. Generate statistics
def get_stats(series, label):
    stats = series.describe(percentiles=[.25, .5, .75, .9, .95, .99])
    print(f"\n--- Distribution for {label} ---")
    print(stats)
    return stats

full_note_stats = get_stats(df['full_note_len'], "Full Note")
conv_stats = get_stats(df['conversation_len'], "Conversation")

# 5. Check against common embedding limits (e.g., 8192 for OpenAI, 512 for BERT)
LIMIT = 8192
print(f"\nExceeding {LIMIT} tokens:")
print(f"Full Note: {(df['full_note_len'] > LIMIT).sum()} rows")
print(f"Conversation: {(df['conversation_len'] > LIMIT).sum()} rows")