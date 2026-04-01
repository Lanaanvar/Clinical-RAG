# Clinical RAG — Medical Case Retrieval System

A Retrieval-Augmented Generation (RAG) system that takes an unstructured patient complaint as input, retrieves similar past clinical cases from a corpus of 30,000 notes, and generates a grounded clinical response.

---

## Architecture

```
Patient Query (plain text)
        ↓
   FastAPI /query
        ↓
  Embed query → Qdrant pre-filter + semantic search
        ↓
  Deduplicate → Fetch full notes from document store
        ↓
  LLM generation (NVIDIA / Llama 3.1 70B)
        ↓
  Grounded clinical response
```

### Small-to-Big Retrieval Strategy

| Layer | Field | Role |
|---|---|---|
| Filter | `summary` | Pre-filter by age, sex |
| Search | `conversation` | Small chunks — precise matching |
| Context | `full_note` | Full document — rich LLM context |

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Vector DB | Qdrant |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | meta/llama-3.1-70b-instruct (NVIDIA Build) |
| Dataset | Vinay393/augmented-clinical-notes (30k rows) |
| Containerization | Docker + Docker Compose |

---

## Project Structure

```
RAG/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── app/
    ├── main.py
    ├── config.py
    ├── ingestion/
    │   ├── ingest.py
    │   ├── chunker.py
    │   ├── document_store.py
    │   ├── indexer.py
    │   └── summary_parser.py
    ├── services/
    │   ├── retriever.py
    │   ├── generator.py
    │   └── rag_pipeline.py
    ├── routes/
    │   └── query.py
    └── models/
        └── schemas.py
```

---

## Quickstart

### 1. Clone and configure
```bash
git clone <your-repo>
cd RAG
cp .env.example .env
# Add your NVIDIA_API_KEY to .env
```

### 2. Start Qdrant
```bash
docker compose up -d qdrant
```

### 3. Run ingestion
```bash
docker compose run --rm ingest
```

### 4. Start the API
```bash
docker compose up -d api
```

### 5. Test
```
http://localhost:8000/docs     ← Swagger UI
http://localhost:8000/health   ← Health check
```

---

## Example Request

```json
POST /query
{
  "patient_query": "severe knee pain for two weeks, worse when walking",
  "filters": {
    "patient_sex": "Male",
    "patient_age_group": "middle_aged"
  },
  "top_k": 3
}
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `NVIDIA_API_KEY` | NVIDIA Build API key |
| `NVIDIA_MODEL` | LLM model name |
| `QDRANT_URL` | Qdrant connection URL |
| `EMBEDDING_MODEL` | Sentence transformer model |
| `DOCUMENT_STORE_PATH` | Path to document store JSON |

See `.env.example` for all variables.

---

## Dataset

**Vinay393/augmented-clinical-notes** — 30,000 clinical case rows with fields: `idx`, `note`, `full_note`, `conversation`, `summary`. License: MIT.
