# sec-rag-demo

**Phase 1 – Demo-grade RAG: Document Ingestion Pipeline**

This repo is the starting point of a phased journey from a bare-minimum demo RAG to a production-grade system. Each phase will intentionally expose you to real problems and trade-offs.

---

## What this phase builds

| Component | Choice | Why / Known limitation |
|-----------|--------|----------------------|
| PDF parsing | PyMuPDF | Fast, good text extraction; **tables come out as plain text** (Phase 2 fix) |
| Chunking | Token-based, 500-800 tokens, 100 overlap | Simple & predictable; **sentence splitter is regex-only** (Phase 2 fix) |
| Embeddings | OpenAI `text-embedding-3-small` | High quality; **every re-ingest costs API $** (Phase 2: cache / dedupe) |
| Vector DB | Weaviate (manual vectors) | Full visibility; **no filtering by metadata yet** (Phase 2 fix) |
| Orchestration | Airflow (LocalExecutor) | Simple; **single-node, no parallelism** (Phase 2 fix) |

---

## Prerequisites

- Docker Desktop running
- Python 3.11+
- An OpenAI API key

---

## Quick start

```bash
# 1. Clone / navigate to this repo
cd sec-rag-demo

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

# ── OR run locally without Airflow ──────────────────────────────
make venv && make install
make ingest
```

---

## Project structure

```
sec-rag-demo/
├── docker-compose.yml        # Weaviate + Airflow stack
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── Makefile                  # Convenience commands
│
├── src/
│   ├── config.py             # All config read from env vars
│   └── ingestion/
│       ├── pdf_parser.py     # PDF → ParsedDocument (page text + metadata)
│       ├── chunker.py        # ParsedDocument → TextChunk list
│       ├── embedder.py       # TextChunk texts → OpenAI vectors
│       ├── weaviate_store.py # Schema creation + batch upsert
│       └── pipeline.py       # Orchestrates all four steps
│
├── dags/
│   └── sec_ingestion_dag.py  # Airflow DAG (4 tasks, manual trigger)
│
└── scripts/
    └── run_ingestion.py      # CLI entrypoint (no Airflow needed)
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

## Chunking strategy

```
min_tokens  = 500
max_tokens  = 800
overlap     = 100   # last ~100 tokens of chunk N become the start of chunk N+1
tokenizer   = cl100k_base  (same as OpenAI embedding models)
```

Chunks are formed greedily by accumulating sentence-split segments until the
next segment would exceed `max_tokens`. The overlap is applied by rewinding the
segment pointer by ~100 tokens before starting the next chunk.

---

## Known Phase-1 limitations (roadmap for Phase 2)

1. **Table extraction** — tables are extracted as raw text. Financial tables
   lose structure. → Phase 2: pdfplumber with table detection.

2. **Regex sentence splitting** — breaks on `.` inside numbers and tickers.
   → Phase 2: spaCy or NLTK sentence tokeniser.

3. **No embedding cache** — every run re-calls the OpenAI API.
   → Phase 2: hash-based deduplication before embedding.

4. **No metadata filtering** — Weaviate queries are pure vector search.
   → Phase 2: add `where` filters on company / quarter / year.

5. **Sequential Airflow** — single LocalExecutor, one task at a time.
   → Phase 2: parallel fan-out per document with dynamic task mapping.

6. **No retrieval / QA layer yet** — this phase only covers ingestion.
   → Phase 2: add a simple retrieval + answer generation module.

---

## Weaviate schema

Collection: `SecDocument`

| Property | Type | Example |
|----------|------|---------|
| company | TEXT | `AAPL` |
| quarter | TEXT | `Q2` |
| year | INT | `2023` |
| source_file | TEXT | `2023 Q2 AAPL.pdf` |
| chunk_index | INT | `0` |
| chunk_text | TEXT | `Apple Inc. reported … ` |
| page_start | INT | `3` |
| page_end | INT | `5` |
| token_count | INT | `612` |
| _vector_ | float[] | `[0.023, -0.11, …]` (1536-dim) |
