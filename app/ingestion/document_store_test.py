"""
test_document_store.py
----------------------
Tests for the DocumentStore covering:
- Build from a mock dataset
- Save and load round-trip
- Lookups (hit, miss, type coercion)
- get_many with partial misses
- Edge cases: empty dataset, missing full_note
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.document_store import DocumentStore


# ── Mock dataset ───────────────────────────────────────────────────────────────

def make_mock_dataset(rows: list[dict]):
    """Minimal iterable that mimics a HuggingFace Dataset."""
    return rows


MOCK_ROWS = [
    {"idx": "155216", "full_note": "A sixteen year old girl presented with neck pain..."},
    {"idx": "77465",  "full_note": "A 56 year old man with chest wall tumor..."},
    {"idx": "133948", "full_note": "A 36 year old female with hip pain..."},
    {"idx": "80176",  "full_note": "A 49 year old male with proximal forearm pain..."},
]


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_build_from_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DocumentStore(path=f"{tmpdir}/store.json")
        store.build_from_dataset(make_mock_dataset(MOCK_ROWS))
        assert len(store) == 4
        print("✅ test_build_from_dataset passed")


def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/store.json"

        # Build and save
        store = DocumentStore(path=path)
        store.build_from_dataset(make_mock_dataset(MOCK_ROWS))
        store.save()

        assert os.path.exists(path)

        # Load fresh instance
        store2 = DocumentStore(path=path)
        store2.load()

        assert len(store2) == 4
        assert store2.get("155216") == MOCK_ROWS[0]["full_note"]
        print("✅ test_save_and_load_roundtrip passed")


def test_get_existing_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DocumentStore(path=f"{tmpdir}/store.json")
        store.build_from_dataset(make_mock_dataset(MOCK_ROWS))

        result = store.get("77465")
        assert result == "A 56 year old man with chest wall tumor..."
        print("✅ test_get_existing_key passed")


def test_get_missing_key_returns_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DocumentStore(path=f"{tmpdir}/store.json")
        store.build_from_dataset(make_mock_dataset(MOCK_ROWS))

        result = store.get("999999")
        assert result is None
        print("✅ test_get_missing_key_returns_none passed")


def test_idx_type_coercion():
    """
    idx coming from Qdrant payload is always a string.
    idx in the dataset might be int or str.
    The store should handle both transparently.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Dataset rows with integer idx
        rows_with_int_idx = [
            {"idx": 155216, "full_note": "Note for integer idx"},
        ]
        store = DocumentStore(path=f"{tmpdir}/store.json")
        store.build_from_dataset(make_mock_dataset(rows_with_int_idx))

        # Lookup with string (as would come from Qdrant payload)
        result = store.get("155216")
        assert result == "Note for integer idx"
        print("✅ test_idx_type_coercion passed")


def test_get_many_all_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DocumentStore(path=f"{tmpdir}/store.json")
        store.build_from_dataset(make_mock_dataset(MOCK_ROWS))

        result = store.get_many(["155216", "77465"])
        assert len(result) == 2
        assert "155216" in result
        assert "77465" in result
        print("✅ test_get_many_all_found passed")


def test_get_many_partial_miss():
    """Missing keys should be silently skipped — not raise an error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DocumentStore(path=f"{tmpdir}/store.json")
        store.build_from_dataset(make_mock_dataset(MOCK_ROWS))

        result = store.get_many(["155216", "DOES_NOT_EXIST"])
        assert len(result) == 1
        assert "155216" in result
        print("✅ test_get_many_partial_miss passed")


def test_get_many_all_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DocumentStore(path=f"{tmpdir}/store.json")
        store.build_from_dataset(make_mock_dataset(MOCK_ROWS))

        result = store.get_many(["MISSING_1", "MISSING_2"])
        assert result == {}
        print("✅ test_get_many_all_missing passed")


def test_skip_rows_with_missing_full_note():
    """Rows with empty or None full_note should be skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rows = [
            {"idx": "100", "full_note": "Valid note"},
            {"idx": "101", "full_note": ""},        # empty — should skip
            {"idx": "102", "full_note": None},      # None — should skip
            {"idx": "103"},                          # missing key — should skip
        ]
        store = DocumentStore(path=f"{tmpdir}/store.json")
        store.build_from_dataset(make_mock_dataset(rows))

        assert len(store) == 1
        assert "100" in store
        assert "101" not in store
        assert "102" not in store
        assert "103" not in store
        print("✅ test_skip_rows_with_missing_full_note passed")


def test_skip_rows_with_missing_idx():
    """Rows with no idx should be silently skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rows = [
            {"full_note": "No idx on this row"},
            {"idx": "", "full_note": "Empty string idx"},
            {"idx": "200", "full_note": "Valid row"},
        ]
        store = DocumentStore(path=f"{tmpdir}/store.json")
        store.build_from_dataset(make_mock_dataset(rows))

        assert len(store) == 1
        assert "200" in store
        print("✅ test_skip_rows_with_missing_idx passed")


def test_empty_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DocumentStore(path=f"{tmpdir}/store.json")
        store.build_from_dataset(make_mock_dataset([]))
        assert len(store) == 0
        print("✅ test_empty_dataset passed")


def test_load_file_not_found():
    """Loading a non-existent file should raise FileNotFoundError."""
    store = DocumentStore(path="/tmp/definitely_does_not_exist_xyz.json")
    try:
        store.load()
        print("❌ test_load_file_not_found FAILED: Expected FileNotFoundError")
    except FileNotFoundError:
        print("✅ test_load_file_not_found passed")


def test_is_loaded_property():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DocumentStore(path=f"{tmpdir}/store.json")
        assert not store.is_loaded

        store.build_from_dataset(make_mock_dataset(MOCK_ROWS))
        assert store.is_loaded
        print("✅ test_is_loaded_property passed")


def test_contains_operator():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DocumentStore(path=f"{tmpdir}/store.json")
        store.build_from_dataset(make_mock_dataset(MOCK_ROWS))

        assert "155216" in store
        assert "999999" not in store
        print("✅ test_contains_operator passed")


def test_saved_json_keys_are_strings():
    """
    JSON serialisation always produces string keys.
    Verify the saved file has string keys so reload is consistent.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/store.json"
        store = DocumentStore(path=path)
        store.build_from_dataset(make_mock_dataset(MOCK_ROWS))
        store.save()

        with open(path) as f:
            raw = json.load(f)

        for key in raw.keys():
            assert isinstance(key, str), f"Expected string key, got {type(key)}: {key}"
        print("✅ test_saved_json_keys_are_strings passed")


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_build_from_dataset,
        test_save_and_load_roundtrip,
        test_get_existing_key,
        test_get_missing_key_returns_none,
        test_idx_type_coercion,
        test_get_many_all_found,
        test_get_many_partial_miss,
        test_get_many_all_missing,
        test_skip_rows_with_missing_full_note,
        test_skip_rows_with_missing_idx,
        test_empty_dataset,
        test_load_file_not_found,
        test_is_loaded_property,
        test_contains_operator,
        test_saved_json_keys_are_strings,
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