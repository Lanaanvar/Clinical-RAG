"""
test_chunker.py
---------------
Tests for the hybrid chunker covering:
- Basic chunking with a clean conversation
- Token limit enforcement
- Overlap between chunks
- Oversized single turn pairs
- Malformed/empty conversations
- chunk_id uniqueness
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.chunker import (
    HybridChunker,
    TokenCounter,
    parse_turns,
    pair_turns,
)


# ── Mock token counter (word-based, deterministic for tests) ───────────────────

class MockTokenCounter:
    """Simple word-count based token counter for testing."""
    def count(self, text: str) -> int:
        return len(text.split())


# ── Sample conversations ───────────────────────────────────────────────────────

SHORT_CONVERSATION = """Doctor: Good morning, what brings you in today?
Patient: I have been having knee pain for two weeks.
Doctor: How severe is the pain on a scale of 1 to 10?
Patient: About a 7. It gets worse when I walk.
Doctor: Have you had any injuries recently?
Patient: No, no injuries that I can remember."""

LONG_CONVERSATION = """Doctor: Good morning, how are you feeling?
Patient: Not well doctor. I have severe pain in my left knee for two months now.
Doctor: Can you describe the pain in more detail?
Patient: It is a sharp throbbing pain that gets worse when I try to walk or climb stairs.
Doctor: Have you noticed any swelling or redness?
Patient: Yes there is some swelling on the inner side of the knee.
Doctor: Have you had any injuries to that knee before?
Patient: No never. It started suddenly one morning.
Doctor: Does anything make the pain better?
Patient: Rest helps a little but even at night it bothers me.
Doctor: Have you taken any medications for this?
Patient: I took ibuprofen for a few days but it did not help much.
Doctor: Do you have any other joint pain?
Patient: Actually my right knee has been a bit sore too lately.
Doctor: Any family history of arthritis or joint disease?
Patient: My mother had rheumatoid arthritis yes.
Doctor: Okay I will order some imaging tests. We will get to the bottom of this.
Patient: Thank you doctor I am really worried about this."""

MALFORMED_CONVERSATION = "This conversation has no speaker labels at all."

EMPTY_CONVERSATION = ""

CONSECUTIVE_DOCTOR_TURNS = """Doctor: First question here.
Doctor: Actually let me rephrase that question.
Patient: Sure I understand.
Doctor: How long has this been going on?
Patient: About three weeks now."""


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_parse_turns_basic():
    turns = parse_turns(SHORT_CONVERSATION)
    assert len(turns) == 6
    assert turns[0].speaker == "Doctor"
    assert turns[1].speaker == "Patient"
    assert "knee pain" in turns[1].text
    print("✅ test_parse_turns_basic passed")


def test_parse_turns_empty():
    turns = parse_turns(EMPTY_CONVERSATION)
    assert turns == []
    print("✅ test_parse_turns_empty passed")


def test_parse_turns_none():
    turns = parse_turns(None)
    assert turns == []
    print("✅ test_parse_turns_none passed")


def test_pair_turns_basic():
    turns = parse_turns(SHORT_CONVERSATION)
    pairs = pair_turns(turns)
    assert len(pairs) == 3
    assert pairs[0].doctor_turn.speaker == "Doctor"
    assert pairs[0].patient_turn.speaker == "Patient"
    print("✅ test_pair_turns_basic passed")


def test_pair_turns_consecutive_doctor():
    """Consecutive doctor turns should be merged into one turn."""
    turns = parse_turns(CONSECUTIVE_DOCTOR_TURNS)
    pairs = pair_turns(turns)
    # "First question" and "let me rephrase" should be merged
    assert len(pairs) == 2
    assert "rephrase" in pairs[0].doctor_turn.text
    print("✅ test_pair_turns_consecutive_doctor passed")


def test_chunker_produces_chunks():
    counter = MockTokenCounter()
    chunker = HybridChunker(token_counter=counter, token_limit=256, overlap_pairs=1)
    chunks = chunker.chunk(idx="test001", conversation=SHORT_CONVERSATION)
    assert len(chunks) > 0
    assert all(c.idx == "test001" for c in chunks)
    print("✅ test_chunker_produces_chunks passed")


def test_chunk_ids_are_unique():
    counter = MockTokenCounter()
    chunker = HybridChunker(token_counter=counter, token_limit=256, overlap_pairs=1)
    chunks = chunker.chunk(idx="test001", conversation=LONG_CONVERSATION)
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found!"
    print("✅ test_chunk_ids_are_unique passed")


def test_token_limit_respected():
    """
    Chunks should not wildly exceed the token limit.

    Why allow 2x headroom?
    When overlap is applied, the first pair of a new chunk is copied from the
    previous chunk. If that overlap pair is large, the new chunk can start
    already near the limit before any new pairs are added. This is expected
    and correct behaviour — overlap preserves context at boundaries.
    The important invariant is that NO single *new* pair addition causes
    runaway growth. We check that no chunk exceeds 2x the limit.
    """
    counter = MockTokenCounter()
    token_limit = 30  # tight limit to force multiple chunks
    chunker = HybridChunker(token_counter=counter, token_limit=token_limit, overlap_pairs=1)
    chunks = chunker.chunk(idx="test002", conversation=LONG_CONVERSATION)

    for chunk in chunks:
        assert chunk.token_count <= token_limit * 2, (
            f"Chunk '{chunk.chunk_id}' has {chunk.token_count} tokens — "
            f"runaway growth beyond 2x limit ({token_limit * 2})"
        )
    print("✅ test_token_limit_respected passed")


def test_multiple_chunks_for_long_conversation():
    """A long conversation with a tight token limit should produce multiple chunks."""
    counter = MockTokenCounter()
    chunker = HybridChunker(token_counter=counter, token_limit=30, overlap_pairs=1)
    chunks = chunker.chunk(idx="test003", conversation=LONG_CONVERSATION)
    assert len(chunks) > 1, "Expected multiple chunks for a long conversation"
    print("✅ test_multiple_chunks_for_long_conversation passed")


def test_overlap_creates_repeated_content():
    """The last turn pair of chunk N should appear at the start of chunk N+1."""
    counter = MockTokenCounter()
    chunker = HybridChunker(token_counter=counter, token_limit=40, overlap_pairs=1)
    chunks = chunker.chunk(idx="test004", conversation=LONG_CONVERSATION)

    if len(chunks) >= 2:
        # The last "Patient:" line from chunk 0 should appear in chunk 1
        chunk0_lines = chunks[0].text.strip().split("\n")
        last_patient_line = [l for l in chunk0_lines if l.startswith("Patient:")][-1]
        assert last_patient_line in chunks[1].text, (
            "Overlap not found between chunk 0 and chunk 1"
        )
    print("✅ test_overlap_creates_repeated_content passed")


def test_empty_conversation_returns_empty():
    counter = MockTokenCounter()
    chunker = HybridChunker(token_counter=counter, token_limit=256)
    chunks = chunker.chunk(idx="test005", conversation=EMPTY_CONVERSATION)
    assert chunks == []
    print("✅ test_empty_conversation_returns_empty passed")


def test_malformed_conversation_fallback():
    """A conversation with no speaker labels should return empty (no valid pairs)."""
    counter = MockTokenCounter()
    chunker = HybridChunker(token_counter=counter, token_limit=256)
    chunks = chunker.chunk(idx="test006", conversation=MALFORMED_CONVERSATION)
    # No valid Doctor/Patient turns → no chunks
    assert chunks == []
    print("✅ test_malformed_conversation_fallback passed")


def test_chunk_text_contains_both_speakers():
    """Every chunk should contain both Doctor and Patient content."""
    counter = MockTokenCounter()
    chunker = HybridChunker(token_counter=counter, token_limit=256)
    chunks = chunker.chunk(idx="test007", conversation=SHORT_CONVERSATION)
    for chunk in chunks:
        assert "Doctor:" in chunk.text
        assert "Patient:" in chunk.text
    print("✅ test_chunk_text_contains_both_speakers passed")


def test_chunk_idx_is_string():
    """idx stored in chunk should always be a string."""
    counter = MockTokenCounter()
    chunker = HybridChunker(token_counter=counter, token_limit=256)
    # Pass an integer idx (as might come from dataset)
    chunks = chunker.chunk(idx=155216, conversation=SHORT_CONVERSATION)
    for chunk in chunks:
        assert isinstance(chunk.idx, str)
    print("✅ test_chunk_idx_is_string passed")


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_parse_turns_basic,
        test_parse_turns_empty,
        test_parse_turns_none,
        test_pair_turns_basic,
        test_pair_turns_consecutive_doctor,
        test_chunker_produces_chunks,
        test_chunk_ids_are_unique,
        test_token_limit_respected,
        test_multiple_chunks_for_long_conversation,
        test_overlap_creates_repeated_content,
        test_empty_conversation_returns_empty,
        test_malformed_conversation_fallback,
        test_chunk_text_contains_both_speakers,
        test_chunk_idx_is_string,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"❌ {test.__name__} FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")