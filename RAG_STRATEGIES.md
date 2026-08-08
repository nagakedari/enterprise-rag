# RAG Strategy Reference

This document catalogs every chunking, embedding, retrieval, reranking, and
evaluation strategy implemented in this repo, and shows how each configuration
scored on the SEC 10-Q golden Q&A benchmark. It is meant to let a reader (or
future me) see the whole design space at a glance and know which combination
actually performed best, without reading every source file.

> Sourced directly from `src/`, `scripts/run_evaluation.py`, `docker-compose.yml`,
> `requirements.txt`, `.env`, and the `evaluation_results*/`, `debug/retrieval/`
> CSV artifacts in this repo. Numbers are means from actual evaluation runs
> (n≈8–68 questions per run — see [Caveats](#caveats--how-to-read-this)).

---

## 1. Chunking strategies

Three chunking strategies are implemented, each with its own config dataclass
(`src/config.py`) and its own Weaviate collection, so all three coexist and can
be A/B tested against the same golden dataset.

| Strategy | Module | Approach | Key parameters |
|---|---|---|---|
| **Basic / token-based** | `src/ingestion/chunker.py` | Regex sentence-split → greedy accumulation up to `max_tokens`, tail-rewind overlap. | `min_tokens=500`, `max_tokens=800`, `overlap_tokens=100`, tokenizer `cl100k_base` |
| **Parent-child** | `src/ingestion/smart_chunker.py` | Split at SEC section headers (ITEM/PART/NOTE) first (no token limit) → `RecursiveCharacterTextSplitter` (separators `["\n\n","\n",". "," ",""]`) within each section builds ~1000-token **parent** chunks → each parent sub-split into ~300-token **child** chunks that get embedded. | `parent_max_tokens=1000` (overlap 100), `child_max_tokens=300` (overlap 50) |
| **Semantic** | `src/ingestion/semantic_chunker.py` | Reuses the same section-boundary split as parent-child, then `SemanticChunker` (LangChain, BGE-M3 embeddings) splits within each section at points where inter-sentence similarity drops. Post-processed with a token ceiling, small-fragment merge, and overlap injection. | `embedding_model=BAAI/bge-m3`, `breakpoint_threshold_type=percentile` @ 85 (bottom 15% similarity), `token_ceiling=512`, `min_char_size=512`, `overlap_chars=256` |

Design notes worth keeping in mind:
- **Section-first, then token-first** was a deliberate fix (documented in
  `evaluation_results_understanding.md`, "Fix 5"): the original parent-child
  chunker used SEC headers *as* splitter separators and hit the token limit
  before finding a header, producing parents that straddled section
  boundaries. Splitting on headers first guarantees no parent crosses a
  section.
- The **semantic chunker's "consistency principle"**: the same BGE-M3 model
  decides *where* to cut (boundary detection) and *how* to embed the chunk for
  search — so split points stay aligned with the retrieval embedding space.
- A parallel **LlamaIndex-native pipeline** (`src/ingestion/llamaindex_pipeline.py`)
  reimplements basic (`SentenceSplitter`) and parent-child
  (`HierarchicalNodeParser` + `AutoMergingRetriever`) chunking using LlamaIndex
  abstractions instead of custom code, kept in separate Weaviate collections
  (`SecDocumentLI`, `SecDocumentSmartLI`) purely for engine-vs-engine A/B
  comparison.

| Chunking strategy | Weaviate collection | Selected via |
|---|---|---|
| basic | `SecDocument` | `--chunking-strategy basic` |
| parent_child | `SecDocumentSmart` | `--chunking-strategy parent_child` |
| semantic | `DocumentChunk` | `--chunking-strategy semantic` |

---

## 2. Embedding models — by pipeline stage

Embedding model choice differs by *stage*, and the semantic-chunking path
intentionally uses a different model family than everything else.

| Stage | Model | Notes |
|---|---|---|
| Ingestion — basic & parent-child chunk embedding | OpenAI `text-embedding-3-small` (1536-dim) | `EmbeddingConfig`, `src/ingestion/embedder.py` |
| Ingestion — semantic chunking (boundary detection **and** chunk embedding) | `BAAI/bge-m3` (local, HuggingFace) | Runs via ONNX Runtime by default (`USE_ONNX=true`, 3–5× faster on CPU than PyTorch), falls back to PyTorch `HuggingFaceEmbeddings` |
| Query-time retrieval — basic & parent-child collections | OpenAI `text-embedding-3-small` | Must match ingestion model to compare in the same vector space |
| Query-time retrieval — semantic collection (`DocumentChunk`) | `BAAI/bge-m3` | Explicitly matched to the ingest-time model (`retriever.py`) |
| LlamaIndex pipeline (ingestion + retrieval) | OpenAI `text-embedding-3-small` via `llama_index.embeddings.openai.OpenAIEmbedding` | Same model as the custom pipeline, for apples-to-apples engine comparison |
| Evaluation — `answer_similarity` metric | OpenAI `text-embedding-3-small` | Cosine similarity between generated vs. golden answer embeddings |
| Evaluation — RAGAS `AnswerCorrectness` (semantic-similarity component) | OpenAI `text-embedding-3-small` | Via `langchain_openai.OpenAIEmbeddings`, called with the *evaluation* API key |

Generation model (context, not an embedding): `gpt-4o-mini`, `temperature=0.0`.

**Judge-model note:** `EvaluationConfig` defaults to `gpt-4o` in code, and the
comment explicitly explains why — *"avoids the 'LLM judging itself'
self-serving bias"* by using a stronger model than the `gpt-4o-mini` generator.
In this repo's actual `.env`, `EVALUATION_MODEL=gpt-4o-mini` overrides that
default, so the recorded evaluation runs use the **same model as both
generator and judge** — worth knowing when interpreting the LLM-scored metrics
below, since self-judging bias is a real risk for `correctness`,
`context_relevance`, `answer_relevance`, and `factual_error_rate`.

---

## 3. Retrieval strategies & tunable parameters

Two retrieval **engines** are implemented in parallel and selectable via
`--engine {custom, llamaindex}`:

- **Custom** (`src/retrieval/retriever.py`) — talks to Weaviate directly.
- **LlamaIndex** (`src/retrieval/llamaindex_retriever.py`) — same behavior via
  LlamaIndex's `VectorStoreIndex` / `AutoMergingRetriever` abstractions,
  against separate collections, for engine-level A/B comparison.

Both support the same two **retrieval modes**:

| Mode | Mechanism |
|---|---|
| `semantic` | Pure cosine-similarity vector search (`near_vector`) |
| `hybrid` | BM25 (lexical) + vector (semantic), fused server-side by Weaviate via Reciprocal Rank Fusion — the default |

### Tunable parameters actually explored

| Parameter | What it controls | Values tried | Where |
|---|---|---|---|
| `alpha` (`RETRIEVAL_HYBRID_ALPHA`) | Hybrid balance: `0.0`=pure BM25, `1.0`=pure vector | 0.1, 0.25, 0.4, 0.5 (default), 0.6, 0.8 | `--alpha` |
| `top_k` | Chunks retrieved per question | 5 (code default), **7 (`.env` override, active default)**, 10 | `--top-k` |
| `rerank_top_k` | Chunks kept *after* reranking (over-fetch `top_k*3` → rerank → keep this many) | 5, 6 | `--rerank-top-k` |
| `filter_mode` | How company/year/quarter are extracted from the question | `llm` (default, one `gpt-4o-mini` call/question), `regex` (curated 18-ticker alias table + ordinal/month/quarter patterns, zero LLM cost) | `--filter-mode` |
| `chunking_strategy` × `retrieval_mode` × `engine` | Which collection/pipeline is queried | basic / parent_child / semantic × semantic / hybrid × custom / llamaindex | `--chunking-strategy`, `--retrieval-mode`, `--engine` |

**Why the alpha tuning happened:** SEC 10-Q questions are heavily
token-specific ("EPS", "Item 7A", ticker symbols, dollar figures) — vocabulary
that BM25 matches exactly but embeddings encode only approximately. This
motivated sweeping alpha toward BM25-heavy values (0.25, 0.1) for the
financial domain (documented reasoning in `evaluation_results_understanding.md`).

**Metadata filtering:** `company` / `year` / `quarter` filters (extracted from
the question) are AND-combined and applied at the Weaviate query level, so a
question about "Apple Q2 2023" won't retrieve Apple Q1 2023 or Q3 2022 chunks
even if they're semantically similar — this was added specifically to fix a
wrong-period retrieval bug (see [Findings](#key-findings-from-the-debugging-notes)).

**Parent-child dedup:** because retrieval scores at the child level but returns
parent text, the retriever over-fetches `top_k * 3` children and deduplicates
by `parent_id`, keeping only the best-scoring child per parent.

---

## 4. Reranking / ranking models

Module: `src/retrieval/reranker.py`. Reranking is optional (`--rerank-mode`)
and runs *after* initial retrieval, on an over-fetched candidate set
(`top_k * 3`), to fix two known weaknesses of vector/BM25 similarity alone:
boilerplate inflation (legal disclaimers score well on both BM25 and vector
similarity) and topic dilution inside large parent chunks.

| Mode | Model | Mechanism | Cost |
|---|---|---|---|
| `llm` | `gpt-4o-mini` (same as generator) | Listwise: one call receives all numbered candidates + question, returns a full relevance ranking. Prompt explicitly tells the model to downweight boilerplate. | ~1 LLM call per retrieval |
| `cross_encoder` | `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers, local) | Pointwise: scores each `(question, chunk)` pair via cross-attention. | Free after install, ~50ms for 15 candidates on CPU |

No Cohere Rerank or other hosted reranker is integrated. Across the recorded
evaluation runs, `cross_encoder` (`ms-marco-MiniLM-L-6-v2`) is the reranker
used in nearly every experiment file; the `llm` rerank mode is implemented and
CLI-selectable but not present in any saved run artifact. Both strategies fall
back to the original retrieval order on any error, so reranking never hard-fails
the pipeline.

---

## 5. Evaluation mechanisms & metrics

Orchestrator: `src/evaluation/evaluator.py` (`evaluate_samples()`), metric
implementations in `src/evaluation/metrics.py`. Ten metrics, grouped by what
they need as input:

| Group | Metric | Source | LLM? |
|---|---|---|---|
| Answer quality (vs. golden reference) | `exactness` | Custom — token-level F1 (bag-of-words, SQuAD-style), no LLM | No |
| | `answer_similarity` | Custom — cosine similarity of OpenAI embeddings | No (embedding call only) |
| | `correctness` | RAGAS `AnswerCorrectness` | Yes |
| Retrieval quality | `context_precision` | RAGAS `ContextPrecision` | Yes |
| | `context_recall` | RAGAS `ContextRecall` | Yes |
| | `context_relevance` | Custom GEval rubric (0/0.25/0.5/0.75/1.0 scale) | Yes |
| Generation grounding | `faithfulness` | RAGAS `Faithfulness` | Yes |
| | `hallucination_rate` | Derived: `1 − faithfulness` | Derived |
| | `answer_relevance` | RAGAS `AnswerRelevancy` | Yes |
| | `factual_error_rate` | Custom GEval — decomposes answer into atomic claims, checks each against the golden answer | Yes |

**RAGAS wiring:** judge LLM = `ChatOpenAI(model=config.evaluation.model)`
(effectively `gpt-4o-mini`, see the judge-model note above); embeddings for
`AnswerCorrectness`'s semantic component = `text-embedding-3-small`; throttled
to `max_workers=4` (down from RAGAS's default 16) because higher concurrency
caused silently-dropped OpenAI calls → NaN metrics.

**Latency is not currently tracked anywhere in the evaluation pipeline** — no
timing metric exists in `ALL_METRICS`, and no evaluation CSV has a latency
column. This is a real gap relative to the "Configuration vs. Latency"
comparison this doc aims for; see [Caveats](#caveats--how-to-read-this).

**Retrieval-only debug mode** (`--debug`, `run_debug_mode()` in
`scripts/run_evaluation.py`) skips generation and metrics entirely and instead
checks whether any retrieved chunk falls on the golden source document's page
range, producing `debug/retrieval/*_retrieval_debug.csv` with a ✅/❌/⚠
per-question hit indicator. This is how alpha values (0.4/0.6/0.8) and other
parameters were spot-checked cheaply before running a full, LLM-scored
evaluation.

**Golden dataset:** SEC 10-Q Q&A pairs from an external `KG-RAG-datasets`
repo, tagged by `Question Type` (`Multi-Doc RAG`, `Single-Doc Multi-Chunk RAG`,
`Single-Doc Single-Chunk RAG`) and `Source Chunk Type` (`Table`, `Text`).

---

## Results by configuration

Each row is one recorded evaluation run (`evaluation_results*_summary.csv` in
the repo). "Configuration" decodes the run-tag naming scheme:
`{engine}_{chunking}_{mode}{_a<alpha>}{_k<top_k>}_{filter}filters{_<rerank>rerank}{_rt<rerank_k>}`.

| Configuration | Context Precision | Context Recall | Faithfulness | Hallucination Rate | Factual Error Rate | Answer Similarity |
|---|--:|--:|--:|--:|--:|--:|
| Phase 1 — basic chunking, semantic-only, no filters/rerank | — | — | 0.38 | 0.62 | 0.74 | 0.46 |
| Phase 2 — parent-child chunking, semantic-only | 0.79 | 0.42 | 0.44 | 0.56 | 0.64 | 0.46 |
| Phase 3 — parent-child + hybrid (RRF) | 0.80 | 0.45 | 0.34 | 0.66 | 0.63 | 0.54 |
| custom · smart(parent-child) · hybrid · no filters/rerank | 0.29 | 0.55 | 0.70 | 0.30 | 0.42 | 0.48 |
| custom · parent_child · hybrid · llm-filters · **no rerank** | 0.55 | 0.60 | 1.00* | — | — | — |
| custom · parent_child · hybrid · llm-filters · **cross-encoder rerank** | 0.43 | 0.70 | 0.54 | 0.46 | 0.32 | 0.53 |
| custom · parent_child · hybrid · **k10** · llm-filters · cross-encoder rerank | 0.59 | 0.61 | 0.43 | 0.57 | 0.47 | 0.57 |
| ↳ same, rerank_top_k=5 (before prompt change) | 0.64 | 0.51 | 0.48 | 0.52 | 0.47 | 0.56 |
| ↳ same, rerank_top_k=6 (**after generation-prompt fix**) | 0.69 | 0.59 | **0.83** | **0.17** | **0.33** | **0.85** |
| custom · semantic(BGE-M3) · hybrid · llm-filters · cross-encoder rerank | 0.20 | 0.32 | 0.44 | 0.56 | 0.50 | 0.58 |
| ↳ same, alpha=0.25 (BM25-heavy) | 0.10 | — | — | — | 0.59 | 0.65 |
| custom · smart · hybrid · alpha=0.25 · **regex-filters** · cross-encoder rerank | 0.31 | 0.75 | 0.50** | 0.50** | 0.47 | 0.62 |
| llamaindex · smart(parent-child) · hybrid | 0.22 | 0.27 | 0.47 | 0.53 | 0.40 | 0.48 |

\* n too small / not all metrics computed in this run (no rerank stage was tested standalone).
\** std=0.71 on this run — high-variance, small sample; treat as noisy.

**Latency:** not measured by any run above — the evaluation pipeline has no
timing instrumentation (see §5). If you want the latency column from a
Semantic/Hybrid/Hybrid+reranker comparison table, it needs to be added to
`run_evaluation.py` (wrap `_retrieve_chunks()` / `generate()` with a timer) —
it doesn't exist in this repo yet.

### Key findings from the debugging notes

`evaluation_results_understanding.md` (untracked working notes in repo root)
diagnoses a specific low-scoring run and drove several of the fixes reflected
in the table above:

1. **Context precision was low (0.29)** because 1000-token parent chunks are
   full of boilerplate (legal disclaimers, table headers) that inflates both
   BM25 and vector similarity scores without being relevant.
2. **Context recall was low (0.55)** because `top_k=5` structurally caps how
   many sections can be retrieved for multi-section questions.
3. **Wrong-period retrieval** — early runs didn't pass company/year/quarter
   filters to the retriever, so a question about "Apple Q2 2023" could pull in
   Apple Q1 2023 or Q3 2022 chunks (semantically near-identical, factually
   different numbers). Fixed by `src/retrieval/query_filters.py`.
4. **Fix 4 (alpha → BM25-heavy)** was reasoned from the observation that SEC
   financial questions hinge on exact tokens (tickers, "Item 7A", dollar
   figures) that BM25 matches better than embeddings — but the recorded
   alpha=0.25 run actually shows **worse** context precision (0.10 vs 0.20 at
   default alpha) on the semantic-chunking collection, suggesting the fix
   didn't generalize the way the hypothesis predicted (or interacted badly
   with BGE-M3 embeddings specifically — this collection uses a different
   embedding model than the parent-child collection).
5. **The rerank_top_k=6 + prompt-change run is the standout result** in the
   table: faithfulness jumped from ~0.43 to 0.83 and hallucination dropped
   from ~0.57 to 0.17, with the *same* retrieval configuration as the row
   above it — this looks like a generation-prompt fix, not a retrieval fix,
   underscoring that faithfulness/hallucination are as sensitive to the
   generation prompt as to retrieval quality.

### Caveats — how to read this

- **Sample sizes vary a lot** across runs (roughly 8 to 68 questions,
  inferred from `debug/retrieval/*.csv` row counts and summary CSV `total`
  columns) and use a fixed random seed, but different `--samples`/`--debug-limit`
  values, so rows are **directionally** comparable, not a controlled
  experiment with matched question sets.
- Several cells are blank because that run computed only a subset of metrics
  (`--metrics ...`) or a metric returned all-NaN (e.g. RAGAS silently failing
  under concurrency).
- Judge-model self-evaluation bias applies to every LLM-scored metric in this
  table (`.env` sets `EVALUATION_MODEL=gpt-4o-mini`, same family as the
  generator) — treat absolute scores as approximate, and relative comparisons
  *within* this table as more trustworthy than the numbers in isolation.
- Latency is absent entirely; add instrumentation before using this table for
  a cost/latency tradeoff decision.
