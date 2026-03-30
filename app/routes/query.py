"""
query.py
--------
FastAPI route handlers for the Clinical RAG API.

Endpoints:
    POST /query   — main RAG endpoint
    GET  /health  — service health check
"""

import logging
import os

from fastapi import APIRouter, HTTPException, Request, status

from models.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrievedCase,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── POST /query ────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the clinical RAG system",
    description=(
        "Submit a plain English patient complaint. "
        "The system retrieves similar past clinical cases and returns "
        "a grounded response based on those cases."
    ),
)
async def query(request: Request, body: QueryRequest):
    """
    Main RAG endpoint.

    - Validates the request body via Pydantic (automatic)
    - Extracts filters if provided
    - Calls the RAG pipeline
    - Returns a structured clinical response
    """
    # ── Validate query is not just whitespace ──────────────────────────────────
    if not body.patient_query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="patient_query cannot be empty or whitespace."
        )

    # ── Extract filters ────────────────────────────────────────────────────────
    filters = body.filters.to_dict() if body.filters else {}

    logger.info(
        f"Query received: '{body.patient_query[:60]}' | "
        f"filters: {filters} | top_k: {body.top_k}"
    )

    # ── Run pipeline ───────────────────────────────────────────────────────────
    try:
        pipeline = request.app.state.pipeline
        result = pipeline.run(
            query=body.patient_query,
            filters=filters,
            top_k=body.top_k,
        )
    except RuntimeError as e:
        # LLM call failed (NVIDIA API error)
        logger.error(f"Pipeline generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM generation failed: {str(e)}"
        )
    except ConnectionError as e:
        # Qdrant is unreachable
        logger.error(f"Qdrant connection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database is unavailable. Please try again later."
        )
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again."
        )

    # ── Build response ─────────────────────────────────────────────────────────
    retrieved_cases = [
        RetrievedCase(
            idx=case["idx"],
            score=case["score"],
            patient_sex=case.get("patient_sex"),
            patient_age_group=case.get("patient_age_group"),
            patient_age=case.get("patient_age"),
            primary_diagnosis=case.get("primary_diagnosis"),
            visit_motivation=case.get("visit_motivation"),
        )
        for case in result["cases"]
    ]

    return QueryResponse(
        answer=result["answer"],
        cases_used=result["cases_used"],
        cases=retrieved_cases,
        query=result["query"],
        model=result["model"],
        retrieval_ms=result["retrieval_ms"],
        generation_ms=result["generation_ms"],
        total_ms=result["total_ms"],
    )


# ── GET /health ────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API and its dependencies are healthy.",
)
async def health(request: Request):
    """
    Health check endpoint.

    Checks:
    - Qdrant is reachable and the collection exists
    - Document store is loaded in memory
    - Returns the embedding model name
    """
    pipeline = request.app.state.pipeline
    retriever = pipeline._retriever

    # ── Check Qdrant ───────────────────────────────────────────────────────────
    qdrant_status = "error"
    try:
        collections = [
            c.name for c in retriever._client.get_collections().collections
        ]
        collection_name = os.getenv("QDRANT_COLLECTION_NAME", "clinical_notes")
        if collection_name in collections:
            qdrant_status = "ok"
        else:
            qdrant_status = f"collection '{collection_name}' not found"
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
        qdrant_status = "unreachable"

    # ── Check document store ───────────────────────────────────────────────────
    doc_store_status = "ok" if retriever._doc_store.is_loaded else "not loaded"

    return HealthResponse(
        status="ok",
        qdrant=qdrant_status,
        document_store=doc_store_status,
        model=os.getenv("EMBEDDING_MODEL", "unknown"),
    )