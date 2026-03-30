"""
summary_parser.py
-----------------
Parses the `summary` field of each dataset row.

The summary field is a JSON string containing structured clinical information.
We extract only the fields that will serve as pre-filters in Qdrant.

Design decisions:
- Always use .get() with defaults — never direct key access
- Return None for missing/null/"None" string values so Qdrant can handle them
- Keep values as simple scalars (str, int) — no nested objects in payload
"""

import json
import re
from typing import Optional


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clean(value) -> Optional[str]:
    """
    Normalise a value extracted from the summary JSON.
    Returns None if the value is missing, the string "None", or empty.
    """
    if value is None:
        return None
    val = str(value).strip()
    if val.lower() in ("none", "", "null", "n/a", "unknown"):
        return None
    return val


# Mapping of English number words to integers (covers typical clinical age descriptions)
_WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "twenty-one": 21, "twenty-two": 22, "twenty-three": 23, "twenty-four": 24,
    "twenty-five": 25, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _parse_age(age_str: Optional[str]) -> Optional[int]:
    """
    Extract a numeric age from strings like:
      "16 years old", "Sixteen years old", "49", "24-day-old"
    Returns None if extraction fails.
    """
    if age_str is None:
        return None

    cleaned = age_str.strip().lower()

    if not cleaned or cleaned in ("none", "null", "n/a", "unknown"):
        return None

    # Infant check — "day-old" or "month-old" → age 0
    if "day" in cleaned or "month" in cleaned:
        return 0

    # Try direct integer string
    if cleaned.isdigit():
        return int(cleaned)

    # Try extracting first Arabic numeral
    match = re.search(r"\d+", cleaned)
    if match:
        return int(match.group())

    # Try matching written-out number words (e.g. "sixteen years old")
    for word, value in _WORD_TO_NUM.items():
        if re.search(rf"\b{word}\b", cleaned):
            return value

    return None


def _age_group(age: Optional[int]) -> Optional[str]:
    """
    Bucket a numeric age into a broad clinical age group.
    This makes pre-filtering more flexible than exact age matching.
    """
    if age is None:
        return None
    if age == 0:
        return "infant"
    if age < 13:
        return "child"
    if age < 18:
        return "adolescent"
    if age < 40:
        return "young_adult"
    if age < 60:
        return "middle_aged"
    return "elderly"


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_summary(summary_raw) -> dict:
    """
    Parse the summary field and return a flat dict of filter-ready metadata.

    Parameters
    ----------
    summary_raw : str or dict
        The raw summary field from the dataset row.

    Returns
    -------
    dict with keys:
        patient_age_raw   : str  | None  — raw age string
        patient_age       : int  | None  — numeric age
        patient_age_group : str  | None  — bucketed age group
        patient_sex       : str  | None  — "Male" | "Female" | None
        visit_motivation  : str  | None  — chief complaint text
        primary_diagnosis : str  | None  — first diagnosis condition if available
    """
    # Default empty result
    result = {
        "patient_age_raw": None,
        "patient_age": None,
        "patient_age_group": None,
        "patient_sex": None,
        "visit_motivation": None,
        "primary_diagnosis": None,
    }

    # ── Step 1: parse the JSON ────────────────────────────────────────────────
    if isinstance(summary_raw, dict):
        data = summary_raw
    elif isinstance(summary_raw, str):
        try:
            data = json.loads(summary_raw)
        except (json.JSONDecodeError, TypeError):
            # Summary is malformed — return empty defaults, don't crash pipeline
            return result
    else:
        return result

    # ── Step 2: patient info ─────────────────────────────────────────────────
    patient_info = data.get("patient information", {}) or {}
    # Some rows have patient information as a list instead of a dict
    # e.g. [{"age": "47", "sex": "Male"}] — take the first element
    if isinstance(patient_info, list):
        patient_info = patient_info[0] if patient_info else {}

    age_raw = _clean(patient_info.get("age"))
    result["patient_age_raw"] = age_raw
    age_numeric = _parse_age(age_raw)
    result["patient_age"] = age_numeric
    result["patient_age_group"] = _age_group(age_numeric)

    sex = _clean(patient_info.get("sex"))
    if sex:
        # Normalise to title case: "male" → "Male"
        result["patient_sex"] = sex.capitalize()

    # ── Step 3: visit motivation ──────────────────────────────────────────────
    visit_motivation = _clean(data.get("visit motivation"))
    result["visit_motivation"] = visit_motivation

    # ── Step 4: primary diagnosis ─────────────────────────────────────────────
    # diagnosis_tests is a list of dicts, each with a "condition" key
    diagnosis_tests = data.get("diagnosis tests", []) or []
    if isinstance(diagnosis_tests, list) and len(diagnosis_tests) > 0:
        first = diagnosis_tests[0] if isinstance(diagnosis_tests[0], dict) else {}
        condition = _clean(first.get("condition"))
        result["primary_diagnosis"] = condition

    return result