"""
generator.py
------------
Calls the NVIDIA-hosted LLM with retrieved clinical context and returns
a grounded clinical response.

NVIDIA Build uses an OpenAI-compatible API — same SDK, different base_url.

Usage:
    generator = Generator()
    response = generator.generate(
        query="I have knee pain when walking",
        notes=["Full clinical note 1...", "Full clinical note 2..."]
    )
    print(response["answer"])
    print(response["cases_used"])  # number of notes passed as context
"""

import logging
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")

# Keep temperature low for factual, grounded clinical responses
# Higher = more creative but less reliable for medical content
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))

# Truncate each note to avoid exceeding context window
# full_notes can be very long — 3000 chars ≈ ~750 tokens per note
# With 5 notes that's ~3750 tokens of context, well within 128k limit
NOTE_MAX_CHARS = int(os.getenv("NOTE_MAX_CHARS", "3000"))


# ── Prompt templates ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical decision support assistant that helps doctors review similar past cases.

Your rules:
- Answer ONLY using information present in the provided clinical cases
- Never add diagnoses, treatments, or facts not mentioned in the cases
- If the provided cases are not relevant to the patient complaint, clearly say so
- Always remind the user that this is for reference only and not a substitute for clinical judgment
- Be concise and structured in your response"""


def build_user_prompt(query: str, notes: list[str]) -> str:
    """
    Build the user prompt by injecting the query and retrieved notes.

    Parameters
    ----------
    query : The patient's plain English complaint
    notes : List of full_note strings retrieved from the document store

    Each note is truncated to NOTE_MAX_CHARS to manage prompt size.
    """
    # Build the cases section
    cases_text = ""
    for i, note in enumerate(notes, start=1):
        truncated = note[:NOTE_MAX_CHARS]
        # Add ellipsis if note was truncated
        if len(note) > NOTE_MAX_CHARS:
            truncated += "\n[... note truncated ...]"
        cases_text += f"\n--- Case {i} ---\n{truncated}\n"

    prompt = f"""Patient complaint:
{query}

Similar past clinical cases retrieved from the knowledge base:
{cases_text}

Based on the above cases, please provide:
1. Possible diagnoses to consider (only those seen in the retrieved cases)
2. Investigations that were performed in similar cases
3. Treatments used in similar cases
4. Any important warnings or red flags noted in the cases

Important: Base your response strictly on the retrieved cases above."""

    return prompt


# ── Generator ──────────────────────────────────────────────────────────────────

class Generator:
    """
    Wraps the NVIDIA-hosted LLM for clinical response generation.

    Initialised once and reused across all queries.
    The OpenAI client is lightweight — no model loading here.
    """

    def __init__(self):
        if not NVIDIA_API_KEY:
            raise ValueError(
                "NVIDIA_API_KEY is not set. "
                "Add it to your .env file: NVIDIA_API_KEY=nvapi-..."
            )

        self._client = OpenAI(
            api_key=NVIDIA_API_KEY,
            base_url=NVIDIA_BASE_URL,
        )
        self._model = NVIDIA_MODEL
        logger.info(f"Generator initialised with model: {self._model}")

    # ── Public interface ───────────────────────────────────────────────────────

    def generate(self, query: str, notes: list[str]) -> dict:
        """
        Generate a grounded clinical response.

        Parameters
        ----------
        query : The patient's plain English complaint
        notes : List of full_note strings from the document store

        Returns
        -------
        {
            "answer"      : str  — the LLM's response
            "cases_used"  : int  — number of notes passed as context
            "model"       : str  — model used
            "query"       : str  — original query echoed back
        }
        """
        if not notes:
            return {
                "answer": (
                    "No relevant clinical cases were found in the knowledge base "
                    "for this query. Please broaden your search or consult clinical "
                    "guidelines directly."
                ),
                "cases_used": 0,
                "model": self._model,
                "query": query,
            }

        # Build prompts
        user_prompt = build_user_prompt(query, notes)

        # Call NVIDIA LLM
        logger.info(
            f"Calling {self._model} with {len(notes)} case(s) as context..."
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            answer = response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"LLM generation failed: {e}") from e

        return {
            "answer": answer,
            "cases_used": len(notes),
            "model": self._model,
            "query": query,
        }


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 60)
    print("GENERATOR — STANDALONE TEST")
    print("=" * 60)

    # Hardcoded fake notes — tests the LLM independently of retrieval
    fake_notes = [
        """A 47-year-old male patient was referred to the rheumatology clinic
because of recurrent attacks of pain in both knees over 1 year.
In September 2016, the patient presented with severe pain over the medial
aspect of the left knee for a two-week duration which prevented him from
ambulation. The pain increased with weight-bearing physical activity.
MRI of the left knee showed a moderate-sized focal area of marrow
edema involving the medial femoral condyle.
The patient was prescribed diclofenac sodium 50mg twice daily and advised
to avoid prolonged weight-bearing activities. Over the next few weeks
the pain subsided and resolved.""",

        """A 36-year-old female patient visited with chief complaint of pain
and restricted range of motion in the left hip joint persisting for two months.
Physical examination revealed severe gait disturbance secondary to hip pain
aggravated by hip joint flexion or rotation.
MRI scan revealed increased joint fluid and bone marrow edema.
The patient underwent Total Hip Arthroplasty after being diagnosed with
idiopathic osteonecrosis of the femoral head and was discharged in good
condition three weeks after surgery.""",
    ]

    fake_query = "I have been having severe knee pain for two weeks, it gets worse when I walk"

    print(f"\nQuery: {fake_query}")
    print(f"Context notes: {len(fake_notes)}")
    print("─" * 60)

    generator = Generator()
    result = generator.generate(query=fake_query, notes=fake_notes)

    print(f"\nModel: {result['model']}")
    print(f"Cases used: {result['cases_used']}")
    print(f"\nResponse:\n")
    print(result["answer"])

    print("\n" + "─" * 60)
    print("Testing with no notes (empty retrieval scenario):")
    empty_result = generator.generate(query=fake_query, notes=[])
    print(empty_result["answer"])

    print("\n" + "=" * 60)
    print("Generator test complete.")
    print("=" * 60)