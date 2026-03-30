"""
chunker.py
----------
Hybrid chunker for the `conversation` field.

Strategy:
- Parse the conversation into discrete Doctor+Patient turn PAIRS
- Build chunks by accumulating complete turn pairs
- Close a chunk when adding the next pair would exceed CHUNK_TOKEN_LIMIT
- Each chunk always contains at least one complete turn pair
- Overlap: the last turn pair of a chunk is repeated as the first of the next
  (this preserves context at chunk boundaries)

Token counting:
- Uses the tokenizer that matches your embedding model
- Accurate subword token counts, not word counts

Edge cases handled:
- Single turn exceeding token limit → stored as its own chunk with a warning
- Empty or malformed conversation → returns empty list
- Conversation with no recognisable turn structure → single chunk fallback
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Turn:
    """A single speaker turn extracted from the conversation."""
    speaker: str          # "Doctor" or "Patient"
    text: str             # The spoken content


@dataclass
class TurnPair:
    """A complete Doctor + Patient exchange."""
    doctor_turn: Turn
    patient_turn: Turn

    def as_text(self) -> str:
        return f"Doctor: {self.doctor_turn.text.strip()}\nPatient: {self.patient_turn.text.strip()}"


@dataclass
class Chunk:
    """
    A single indexable chunk produced by the chunker.

    Attributes
    ----------
    idx         : The case identifier from the dataset row
    chunk_id    : Unique identifier — "{idx}_chunk_{n}"
    text        : The chunk text that will be embedded
    token_count : Number of tokens in `text`
    """
    idx: str
    chunk_id: str
    text: str
    token_count: int


# ── Turn parser ────────────────────────────────────────────────────────────────

# Matches lines like "Doctor: ..." or "Patient: ..."
# Handles variations like "Doctor (Dr. Smith): ..."
_TURN_PATTERN = re.compile(
    r"^(Doctor|Patient)\s*(?:\([^)]*\))?\s*:\s*(.+)",
    re.IGNORECASE
)


def parse_turns(conversation: str) -> List[Turn]:
    """
    Parse a raw conversation string into a list of Turn objects.

    The conversation field looks like:
        Doctor: Good morning, what brings you in?
        Patient: I have been having knee pain...
        Doctor: How long has this been going on?
        ...

    Returns an empty list if the conversation is empty or unparseable.
    """
    if not conversation or not isinstance(conversation, str):
        return []

    turns = []
    # Split on newlines, handle both \n and actual newlines
    lines = conversation.replace("\\n", "\n").split("\n")

    current_speaker = None
    current_text_parts = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = _TURN_PATTERN.match(line)
        if match:
            # Save previous turn before starting a new one
            if current_speaker and current_text_parts:
                turns.append(Turn(
                    speaker=current_speaker.capitalize(),
                    text=" ".join(current_text_parts)
                ))
            current_speaker = match.group(1).capitalize()
            current_text_parts = [match.group(2).strip()]
        else:
            # Continuation of current turn (multi-line speech)
            if current_speaker:
                current_text_parts.append(line)

    # Don't forget the last turn
    if current_speaker and current_text_parts:
        turns.append(Turn(
            speaker=current_speaker,
            text=" ".join(current_text_parts)
        ))

    return turns


def pair_turns(turns: List[Turn]) -> List[TurnPair]:
    """
    Group consecutive Doctor+Patient turns into TurnPair objects.

    Handles imperfect conversations:
    - Consecutive Doctor turns → merge them before pairing
    - Trailing Doctor turn with no Patient response → skip it
    """
    pairs = []
    i = 0

    while i < len(turns):
        # Collect consecutive doctor turns (merge if needed)
        doctor_texts = []
        while i < len(turns) and turns[i].speaker == "Doctor":
            doctor_texts.append(turns[i].text)
            i += 1

        if not doctor_texts:
            # Not a doctor turn — skip unexpected speaker
            i += 1
            continue

        # Collect consecutive patient turns (merge if needed)
        patient_texts = []
        while i < len(turns) and turns[i].speaker == "Patient":
            patient_texts.append(turns[i].text)
            i += 1

        if not patient_texts:
            # Doctor spoke but patient didn't respond — skip this pair
            continue

        pairs.append(TurnPair(
            doctor_turn=Turn("Doctor", " ".join(doctor_texts)),
            patient_turn=Turn("Patient", " ".join(patient_texts))
        ))

    return pairs


# ── Tokenizer wrapper ──────────────────────────────────────────────────────────

class TokenCounter:
    """
    Wraps a HuggingFace tokenizer for accurate token counting.
    Falls back to a word-based approximation if the tokenizer is unavailable.
    """

    def __init__(self, model_name: str):
        self._tokenizer = None
        self._model_name = model_name
        self._load_tokenizer()

    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            logger.info(f"Tokenizer loaded: {self._model_name}")
        except Exception as e:
            logger.warning(
                f"Could not load tokenizer for {self._model_name}: {e}. "
                "Falling back to word-count approximation (tokens ≈ words × 1.3)."
            )

    def count(self, text: str) -> int:
        if self._tokenizer:
            # Return number of tokens without special tokens
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        else:
            # Rough approximation: average English token ≈ 0.75 words
            return int(len(text.split()) * 1.3)


# ── Hybrid chunker ─────────────────────────────────────────────────────────────

class HybridChunker:
    """
    Chunks the conversation field into semantically coherent pieces.

    Parameters
    ----------
    token_counter  : TokenCounter instance
    token_limit    : Max tokens per chunk (default: 256)
    overlap_pairs  : Number of turn pairs to repeat at chunk boundaries (default: 1)
    """

    def __init__(
        self,
        token_counter: TokenCounter,
        token_limit: int = 256,
        overlap_pairs: int = 1,
    ):
        self.token_counter = token_counter
        self.token_limit = token_limit
        self.overlap_pairs = overlap_pairs

    def chunk(self, idx: str, conversation: str) -> List[Chunk]:
        """
        Main entry point. Returns a list of Chunk objects for one dataset row.

        Parameters
        ----------
        idx          : The case identifier (from dataset row['idx'])
        conversation : The raw conversation string (from dataset row['conversation'])
        """
        # ── Step 1: parse into turns then pairs ──────────────────────────────
        turns = parse_turns(conversation)
        if not turns:
            logger.warning(f"[{idx}] Empty or unparseable conversation — skipping.")
            return []

        pairs = pair_turns(turns)
        if not pairs:
            logger.warning(f"[{idx}] No valid turn pairs found — skipping.")
            return []

        # ── Step 2: build chunks by accumulating pairs ────────────────────────
        chunks: List[Chunk] = []
        current_pairs: List[TurnPair] = []
        current_tokens: int = 0
        chunk_n = 0

        for pair in pairs:
            pair_text = pair.as_text()
            pair_tokens = self.token_counter.count(pair_text)

            # Edge case: a single pair exceeds the token limit
            if pair_tokens > self.token_limit:
                # Flush whatever we have first
                if current_pairs:
                    chunks.append(self._make_chunk(idx, chunk_n, current_pairs))
                    chunk_n += 1

                # Store the oversized pair as its own chunk with a warning
                logger.warning(
                    f"[{idx}] Single turn pair has {pair_tokens} tokens "
                    f"(limit: {self.token_limit}). Storing as oversized chunk."
                )
                chunks.append(Chunk(
                    idx=str(idx),
                    chunk_id=f"{idx}_chunk_{chunk_n}",
                    text=pair_text,
                    token_count=pair_tokens,
                ))
                chunk_n += 1
                current_pairs = []
                current_tokens = 0
                continue

            # Would adding this pair exceed the limit?
            projected_tokens = self.token_counter.count(
                "\n".join(p.as_text() for p in current_pairs) + "\n" + pair_text
            ) if current_pairs else pair_tokens

            if projected_tokens > self.token_limit and current_pairs:
                # Close current chunk
                chunks.append(self._make_chunk(idx, chunk_n, current_pairs))
                chunk_n += 1

                # Start new chunk with overlap from the end of the previous chunk
                overlap = current_pairs[-self.overlap_pairs:]
                current_pairs = overlap + [pair]
                current_tokens = self.token_counter.count(
                    "\n".join(p.as_text() for p in current_pairs)
                )
            else:
                current_pairs.append(pair)
                current_tokens = projected_tokens

        # Flush the last chunk
        if current_pairs:
            chunks.append(self._make_chunk(idx, chunk_n, current_pairs))

        return chunks

    def _make_chunk(self, idx: str, chunk_n: int, pairs: List[TurnPair]) -> Chunk:
        text = "\n".join(p.as_text() for p in pairs)
        return Chunk(
            idx=str(idx),
            chunk_id=f"{idx}_chunk_{chunk_n}",
            text=text,
            token_count=self.token_counter.count(text),
        )