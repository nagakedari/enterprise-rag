# Enterprise RAG — SEC 10-Q Filing Q&A

A retrieval-augmented question-answering system over SEC 10-Q quarterly
filings. What started as a bare-minimum demo pipeline (PDF → fixed-size
chunks → OpenAI embeddings → Weaviate) has grown into a system with multiple
interchangeable chunking, retrieval, reranking, and diversity-selection
strategies, a full RAGAS + custom-metric evaluation harness, and a web UI for
running and comparing evaluations.

**This README covers what the project is and how to run it.** For a deep
dive into *which* chunking/retrieval/reranking strategy performs best and
why — including actual evaluation numbers — see
**[RAG_STRATEGIES.md](RAG_STRATEGIES.md)**.

---

## What's in here

- **Ingestion** — PDF parsing (PyMuPDF), three interchangeable chunking
  strategies, OpenAI + local (BGE-M3) embeddings, Weaviate storage.
  Orchestrated via Airflow or run directly as a CLI.
- **Retrieval** — semantic and hybrid (BM25 + vector) search, two engines
  (a custom Weaviate client and a LlamaIndex-based implementation), optional
  reranking (LLM or local cross-encoder), and post-rerank diversity selection
  (MMR / metadata-slot / source-cap) for multi-filing questions.
- **Generation** — OpenAI chat completion over retrieved context.
- **Evaluation** — a CLI (`scripts/run_evaluation.py`) that runs the full
  pipeline against a golden Q&A dataset and scores it with 10 metrics (RAGAS
  + custom GEval), plus a **FastAPI backend + React UI** for picking an
  evaluation configuration, checking whether it's already been run, and
  launching/watching/canceling new runs.

---

## Prerequisites

- Docker Desktop running (for Weaviate + Airflow)
- Python 3.11+
- Node.js 18+ (only needed for the evaluation web UI)
- An OpenAI API key

---

## Quick start

```bash
# 1. Clone / navigate to this repo
cd enterprise-rag

# 2. Create your .env
make env
# Then edit .env and set OPENAI_API_KEY=sk-...

# 3. Link the SEC document corpus into ./data/docs
make link-docs

# 4. Start Weaviate + Airflow
make up
# Airflow UI: http://localhost:8081  (admin / admin)
# Weaviate:   http://localhost:8080

# 5. Trigger the ingestion DAG from the UI — or via CLI:
make trigger

# ── OR run ingestion locally without Airflow ────────────────────
make venv && make install
make ingest
```

### Running an evaluation

```bash
# CLI — see --help for every chunking/retrieval/reranking/diversity flag
.venv/bin/python scripts/run_evaluation.py --samples 25

# OR the web UI (pick parameters, check for existing results, run/re-run,
# watch live progress) — needs Weaviate running (`make up`) and OPENAI_API_KEY set
make api          # FastAPI backend on :8000
make ui-install   # first time only
make ui           # React dev server on :5173
```

See [RAG_STRATEGIES.md](RAG_STRATEGIES.md) for what every flag/parameter
actually does and how different configurations have scored.

---

## Project structure

```
enterprise-rag/
├── docker-compose.yml        # Weaviate + Airflow stack
├── requirements.txt          # Python dependencies (backend + ingestion + eval)
├── .env.example               # Environment variable template
├── Makefile                  # Convenience commands (see `make help`)
├── RAG_STRATEGIES.md          # Strategy deep-dive + evaluation results
│
├── src/
│   ├── config.py              # All config, read from env vars
│   ├── ingestion/
│   │   ├── pdf_parser.py       # PDF → ParsedDocument (page text + metadata)
│   │   ├── chunker.py          # Basic token-based chunking
│   │   ├── smart_chunker.py    # Parent-child chunking (section-aware)
│   │   ├── semantic_chunker.py # Semantic chunking (BGE-M3 boundaries)
│   │   ├── llamaindex_pipeline.py  # LlamaIndex-native ingestion pipeline
│   │   ├── embedder.py         # OpenAI embedding calls
│   │   ├── weaviate_store.py   # Schema creation + batch upsert
│   │   └── pipeline.py         # Orchestrates the custom ingestion pipeline
│   ├── retrieval/
│   │   ├── retriever.py         # Custom Weaviate retriever (+ per-filing retrieval)
│   │   ├── llamaindex_retriever.py  # LlamaIndex-based retriever
│   │   ├── reranker.py          # LLM / cross-encoder reranking + diversity selection
│   │   └── query_filters.py     # Company/year/quarter extraction from a question
│   ├── generation/
│   │   └── generator.py         # Answer generation from retrieved chunks
│   ├── evaluation/
│   │   ├── evaluator.py         # Orchestrates all 10 metrics (RAGAS + custom)
│   │   ├── metrics.py           # Custom metric implementations
│   │   └── run_tag.py           # Shared run-tag naming, used by CLI + API
│   └── api/
│       ├── main.py              # FastAPI app (chat endpoint + evaluation router)
│       ├── models.py            # Pydantic request/response models
│       ├── evaluation_routes.py # Evaluation HTTP endpoints
│       └── evaluation_service.py # History scan/lookup + subprocess job runner
│
├── frontend/                  # React + TypeScript evaluation UI (Vite)
│
├── dags/
│   └── sec_ingestion_dag.py   # Airflow DAG (4 tasks, manual trigger)
│
└── scripts/
    ├── run_ingestion.py        # Ingestion CLI entrypoint (no Airflow needed)
    ├── run_evaluation.py       # Evaluation CLI entrypoint
    └── inspect_weaviate.py     # Quick Weaviate collection inspector
```

---

## Airflow DAG tasks

```
validate_docs → parse_and_chunk → embed_and_store → verify_ingestion
```

| Task | What it does |
|------|-------------|
| `validate_docs` | Confirms PDF files exist before spending API credits |
| `parse_and_chunk` | Parses PDFs and counts chunks (dry run, no API calls) |
| `embed_and_store` | Runs the full pipeline: parse → chunk → embed → Weaviate |
| `verify_ingestion` | Queries Weaviate to confirm objects were written |

---

## Weaviate collections

Each chunking strategy and engine writes to its own collection, so they
coexist and can be evaluated side by side (see
[RAG_STRATEGIES.md](RAG_STRATEGIES.md) for how they compare):

| Collection | Written by |
|---|---|
| `SecDocument` | Custom pipeline, basic chunking |
| `SecDocumentSmart` | Custom pipeline, parent-child chunking |
| `DocumentChunk` | Custom pipeline, semantic chunking |
| `SecDocumentLI` / `SecDocumentSmartLI` | LlamaIndex-native pipeline (basic / parent-child) |

Inspect any collection with `.venv/bin/python scripts/inspect_weaviate.py`.

---

## Known limitations

- Evaluation-run latency isn't tracked anywhere in the pipeline.
- The evaluation web UI's job tracking is in-memory only — it doesn't survive
  a backend restart (results on disk are unaffected; only live-progress state
  is lost). Don't run the API with `--reload` or multiple workers while a job
  is active.
- `DOCS_PATH` / the golden Q&A dataset path currently point at a corpus
  outside this repo — adjust `.env` and `scripts/run_evaluation.py --input`
  for your own dataset.

See [RAG_STRATEGIES.md's Caveats section](RAG_STRATEGIES.md#caveats--how-to-read-this)
for evaluation-methodology caveats (sample sizes, judge-model choice, etc.).
