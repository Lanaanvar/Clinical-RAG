"""
main.py
-------
FastAPI application entry point.

Responsibilities:
- Create the FastAPI app instance
- Handle startup (initialise RAG pipeline) and shutdown via lifespan
- Register routes
- Configure logging

Run with:
    cd app
    uvicorn main:app --reload --port 8000

Then open:
    http://localhost:8000/docs      ← Swagger UI
    http://localhost:8000/health    ← Health check
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

# Ensure app/ is on the path so all internal imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from routes.query import router
from services.rag_pipeline import RAGPipeline

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup and once at shutdown.

    Startup:
        - Initialise RAGPipeline (loads embedding model, connects to Qdrant,
          loads document store into memory)
        - Store on app.state so all routes can access it

    Shutdown:
        - Clean up if needed (currently nothing to clean up)

    Why lifespan instead of @app.on_event("startup")?
        lifespan is the modern FastAPI pattern. on_event is deprecated.
    """
    # ── Startup ────────────────────────────────────────────────────────────────
    logger.info("Starting up Clinical RAG API...")
    logger.info("Initialising RAG pipeline — this may take a few seconds...")

    try:
        app.state.pipeline = RAGPipeline()
        logger.info("RAG pipeline ready. API is accepting requests.")
    except Exception as e:
        logger.error(f"Failed to initialise RAG pipeline: {e}")
        # Re-raise so FastAPI knows startup failed
        raise

    yield  # ← API is live and serving requests here

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("Shutting down Clinical RAG API...")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=os.getenv("APP_TITLE", "Clinical RAG API"),
    version=os.getenv("APP_VERSION", "0.1.0"),
    description=(
        "A Retrieval-Augmented Generation system for clinical decision support. "
        "Submit a patient complaint in plain English and receive a grounded "
        "response based on similar past clinical cases."
    ),
    lifespan=lifespan,
)

# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(router)