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
self-serving bias"* by using a stronger model than the generator. `.env` sets
`GENERATION_MODEL` unset (code default `gpt-4o-mini`) and
`EVALUATION_MODEL=gpt-4o` explicitly, so generation and judging are now
genuinely **different models** — the self-judging bias risk this design was
meant to avoid no longer applies to `correctness`, `context_relevance`,
`answer_relevance`, or `factual_error_rate`. (Earlier revisions of this repo
had `.env` overriding `EVALUATION_MODEL` down to `gpt-4o-mini`, which
collapsed generator and judge into the same model — that override has since
been removed.)

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
| `diversity_mode`, `mmr_lambda`, `per_filing`, `chunks_per_filing`, `max_per_entity` | Post-rerank chunk-selection strategy and per-filing retrieval — see [§5 Diversity selection strategies](#5-diversity-selection-strategies) and [§7 Fix 2](#7-fix-2--per-filing-retrieval-for-temporal-questions) | `none` (default), `mmr` (λ 0.4–0.5 tried), `metadata_slots`, `source_cap`; `per_filing` on/off | `--diversity-mode`, `--mmr-lambda`, `--per-filing`, `--chunks-per-filing`, `--max-per-entity` |

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

## 5. Diversity selection strategies

**Problem addressed:** `context_recall=0.57` — the cross-encoder reranker
optimises individual chunk relevance (pointwise, no awareness of the already-chosen
set), so on Multi-Doc questions it may fill all 6 `rerank_top_k` slots with
chunks from one filing (the highest-scoring one) and leave the second filing
entirely unrepresented. The generated answer is then faithful to what it
received but factually incomplete: facts from the missing filing appear in the
golden answer, so `context_recall` drops and `factual_error_rate` rises.

After reranker scoring, the final `top_k` selection step is replaceable via
`--diversity-mode`. Three alternatives to plain top-k sort are implemented in
`src/retrieval/reranker.py`:

### 5.1 MMR — Maximal Marginal Relevance (`--diversity-mode mmr`)

**How it works:** Iterative greedy selection. At each step pick the candidate
that maximises `λ × norm_relevance − (1−λ) × max_cosine_sim(candidate, selected)`.
`λ=1.0` degenerates to plain top-k; `λ=0.0` is pure diversity; `λ=0.5` (default)
balances both. TF-IDF cosine similarity between chunk texts measures redundancy.
Requires `scikit-learn`.

**Why:** Prevents the reranker from picking four copies of the same balance-sheet
table row. A diverse set of chunks covers more facts, improving `context_recall`
at the cost of some individual-chunk precision.

**Known limitation:** MMR is purely text-diversity aware — it does not guarantee
that specific (company, quarter, year) entities are represented. It also can
pull in historically-adjacent chunks (e.g. a ZeniMax acquisition description
that appeared as background context in a Q2 2023 filing) which the LLM may then
report as a current-period finding.

**Results (n=25, `a0.25_k10_rt6`):**

| Metric | Baseline (no diversity) | MMR λ=0.4 | Δ |
|---|--:|--:|--:|
| context_recall | 0.565 | **0.633** | +0.068 |
| context_precision | 0.346 | 0.399 | +0.053 |
| faithfulness | 0.871 | **0.933** | +0.062 |
| hallucination_rate | 0.129 | **0.067** | −0.062 |
| factual_error_rate | 0.520 | 0.550 | +0.030 |

Recall improved (+6.8pp) and faithfulness jumped (+6.2pp), but `factual_error_rate`
slightly worsened — the diverse-but-temporal-mismatch problem (see Fix 1 below).

### 5.2 Metadata slots — proportional floor allocation (`--diversity-mode metadata_slots`)

**How it works:** Groups candidates by `(company, quarter, year)` entity. Each
entity gets `floor(top_k / n_entities)` guaranteed slots filled by its
highest-scoring chunks (FLOOR / minimum guarantee). Remaining slots go to
globally highest-scoring unchosen chunks.

**Why:** Directly targets Multi-Doc questions where two or more filings must
both be represented. Metadata-slot allocation is entity-aware, not just
text-diversity-aware, so it guarantees that Apple Q2 2023 AND Apple Q3 2023
each get at least one slot even when one quarter scores universally higher.

**Complements MMR:** MMR is text-diversity (reduces duplicate table rows);
metadata_slots is entity-diversity (guarantees filing coverage). They attack
from different angles.

**Results (n=25, `a0.25_k10_rt6`):**

| Metric | Baseline (no diversity) | Metadata slots | Δ |
|---|--:|--:|--:|
| context_recall | 0.565 | 0.549 | −0.016 |
| context_precision | 0.346 | 0.358 | +0.012 |
| faithfulness | 0.871 | **0.926** | +0.055 |
| hallucination_rate | 0.129 | **0.074** | −0.055 |
| factual_error_rate | 0.520 | **0.540** | +0.020 |

Faithfulness improved (+5.5pp) but recall barely moved; metadata_slots is more
conservative than MMR — it still fills "extra" slots globally, so if the
globally-best chunks are all from one filing it still dominates after the floor
is met.

### 5.3 Source cap — per-entity upper bound (`--diversity-mode source_cap`)

**How it works:** Greedy pass over scored candidates (best first). Each
`(company, quarter, year)` entity may claim at most `cap` slots; deferred chunks
fill any remaining slots. Default `cap = ceil(top_k / n_entities)`, minimum 2.
`--max-per-entity N` overrides the auto-cap.

**Why:** Metadata_slots is a FLOOR (ensures a minimum). Source_cap is a CAP
(prevents a monopoly). Together they bracket the slot count from both sides.
Use source_cap when you want to prevent one high-scoring filing from consuming
all slots; use metadata_slots when you want to guarantee the low-scoring filing
gets at least one slot.

**When most useful:** Multi-Doc questions where the question is phrased globally
(no quarter filter) but one period dominates vector similarity — source_cap
forces the cross-encoder to look at other periods.

**Status:** Implemented (n=25 evaluation run pending). Selectable via
`--diversity-mode source_cap` and the React UI "Diversity mode" dropdown.

---

## 6. Fix 1 — Temporal anchoring in the generation prompt

**File:** `src/generation/generator.py` (`_SYSTEM_PROMPT`)

**Problem diagnosed:** `factual_error_rate=0.55` (21/25 flagged, GEval
claim audit). Root-cause breakdown of the 21 flagged questions:

| Root cause | Count | Symptoms |
|---|---|---|
| Retrieval failure (context_precision=0, faithfulness=1.0) | ~7 | LLM faithfully reports wrong chunks — FER=1.0 is a retrieval problem, not a generation problem |
| MMR temporal over-reporting | ~4 | MMR pulled in historically-adjacent chunks from prior periods; LLM (per Rule 1 "READ ALL SOURCES") reported them as primary findings |
| GEval vs. analytical answers | ~10 | Valid interpretive answers that diverge from the golden reference style |

**Specific case driving Fix 1:** "What acquisitions did Microsoft complete in
Q2 2023?" — MMR retrieved a ZeniMax chunk (2021 historical background that
appeared in the Q2 2023 filing as context). The LLM, following Rule 1
("READ ALL SOURCES FIRST"), reported ZeniMax as a Q2 2023 acquisition.
`context_recall=1.0` (golden facts were retrieved) but `factual_error_rate=1.0`
(wrong period claim). Faithfulness was also 1.0 — the answer was perfectly
grounded in the retrieved context but the context itself contained a trap.

**Fix:** Added Rule 5 `TEMPORAL ANCHORING` between Rules 4 and 6 in
`_SYSTEM_PROMPT`:

```
5. TEMPORAL ANCHORING — confine findings to the period the question asks about.
   If the question specifies a particular quarter or year (e.g. "Q2 2023",
   "fiscal 2022"), treat only information from that period as primary findings.
   Facts from other periods that appear in the sources are background context —
   do not elevate them to primary claims in your answer. If a source mentions an
   event from an earlier period as historical background, label it as such rather
   than presenting it as a direct answer to the question.
   Exception: if the question explicitly asks about multiple periods or trends
   over time, cover all relevant periods.
```

**Expected impact:** Directly targets the ~4 temporal over-reporting cases.
Does not hurt faithfulness (the LLM still cites all sources; it just labels
off-period facts as background). Neutral to the ~7 retrieval-failure cases
and the ~10 analytical-answer cases.

**Status:** Implemented. First post-fix run (`..._rt6_mmr_l0.4_pf`, timestamped
right after this prompt change) shows faithfulness at 0.938 and hallucination
at 0.062 — the best of any recorded run — but that run also combines MMR and
per-filing retrieval simultaneously, so the improvement can't be isolated to
the prompt change alone. A clean A/B (same retrieval config, prompt-only
toggle) is still pending.

---

## 7. Fix 2 — Per-filing retrieval for temporal questions

**Files:** `src/retrieval/retriever.py`, `scripts/run_evaluation.py`

**Problem addressed:** For questions like "How has Apple's net sales changed
over time?", `extract_query_filters()` correctly returns `quarter=None` (no
quarter constraint). But a single global hybrid query with `quarter=None` still
tends to return chunks from the filing that scored highest overall — often just
one period. The model then reports one quarter's data, producing low
`context_recall` and high `factual_error_rate` for trend questions.

**Approach:** For questions with `company ≠ None` AND `quarter = None`, run N
separate retrieval queries — one per `(year, quarter)` pair discovered in the
collection — then merge and deduplicate results by `(source_file, chunk_index)`.

**Two new functions in `src/retrieval/retriever.py`:**

- `_discover_filings(collection, company)` — fetches up to 500 objects
  filtered to `company`, extracts unique `(year, quarter)` pairs from their
  metadata. Runs once per question; result is ~20 rows for a typical company
  with 4 years of quarterly filings.

- `retrieve_per_filing(query, config, company, chunks_per_filing=3, ...)` —
  embeds the query once, discovers all filings, runs one hybrid/near_vector
  query per `(year, quarter)` with `chunks_per_filing` results, merges and
  deduplicates. Total candidate pool = `n_filings × chunks_per_filing` before
  reranking.

**Routing logic** (`_retrieve_chunks()` in `scripts/run_evaluation.py`):

```python
should_per_filing = (
    per_filing                              # flag must be set
    and filters.get("quarter") is None      # question has no quarter constraint
    and filters.get("company") is not None  # must know which company
    and engine != "llamaindex"              # not supported for LlamaIndex engine
)
```

**Trade-off:** N Weaviate queries instead of 1 (N ≈ 8–20 for companies with
2–5 years of quarterly data). Each query is lightweight (fetch_limit=9 for
`chunks_per_filing=3` with 3× over-fetch). Total latency increase ≈ N × single
query time, which should be acceptable for evaluation runs but would need
caching for a production API.

**Complements source_cap:** Per-filing guarantees that every period enters the
candidate pool. Source_cap then enforces an upper bound so no single period
monopolises the final `rerank_top_k` slots after the cross-encoder scores them.

**Status:** Implemented. Selectable via `--per-filing` / `--chunks-per-filing`
CLI flags and the React UI "Per-filing retrieval" checkbox. One run so far
combines it with MMR (`..._rt6_mmr_l0.4_pf`, see the results table) — context_recall
0.628 and correctness 0.582 (best correctness of any recorded run), roughly on
par with MMR alone on recall/faithfulness. An isolated per-filing-only run
(no MMR) to separate its individual effect is still pending.

---

## 8. Evaluation mechanisms & metrics

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
(`gpt-4o` per current `.env`, see the judge-model note above — a different,
stronger model than the `gpt-4o-mini` generator); embeddings for
`AnswerCorrectness`'s semantic component = `text-embedding-3-small`; throttled
to `max_workers=4` (down from RAGAS's default 16) because higher concurrency
caused silently-dropped OpenAI calls → NaN metrics.

**Latency is not currently tracked anywhere in the evaluation pipeline** — no
timing metric exists in `ALL_METRICS`, and no evaluation CSV has a latency
column. This is a real gap relative to the "Configuration vs. Latency"
comparison this doc aims for; see [Caveats](#caveats--how-to-read-this).

**Summary CSVs are now self-documenting.** Newer summary CSVs (the diversity-mode
and per-filing runs onward) append three extra `_meta_*` rows after the metric
rows: `_meta_input_file` (which golden Q&A CSV the run used), `_meta_run_tag`,
and `_meta_n_questions` — stamped so a summary file alone answers "which
dataset produced this?" without cross-referencing the run command. Older
summary CSVs (the historical-phase baselines and the pre-diversity runs) don't
have these rows.

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

Each row is one recorded evaluation run (`evaluation_results*_summary.csv`).
Numbers are means from the actual CSV files (n=25 for all recent runs, fixed
seed=42). Run-tag naming: `{engine}_{chunking}_{mode}{_a<alpha>}{_k<top_k>}_{filter}filters{_<rerank>rerank}{_rt<rerank_k>}{_<diversity>}`.

### Historical phase baselines (from earlier project phases, pre-current-pipeline)

| Configuration | Ctx Prec | Ctx Recall | Faithfulness | Hall. Rate | FER | Ans. Sim |
|---|--:|--:|--:|--:|--:|--:|
| Phase 1 — basic, semantic, no filters/rerank | — | — | 0.38 | 0.62 | 0.74 | 0.46 |
| Phase 2 — parent-child, semantic | 0.79 | 0.42 | 0.44 | 0.56 | 0.64 | 0.46 |
| Phase 3 — parent-child + hybrid (RRF) | 0.80 | 0.45 | 0.34 | 0.66 | 0.63 | 0.54 |

### Current pipeline runs (n=25, actual CSV numbers)

| Run tag | Ctx Prec | Ctx Recall | Faithfulness | Hall. Rate | FER | Correctness |
|---|--:|--:|--:|--:|--:|--:|
| `custom_basic_semantic_llmfilters` (baseline, no rerank) | 0.353 | 0.574 | 0.841 | 0.159 | 0.640 | 0.531 |
| `custom_parent_child_semantic_k10_llmfilters` (semantic retrieval) | **0.472** | 0.620 | 0.894 | 0.105 | 0.500 | **0.578** |
| `custom_parent_child_hybrid_a0.25_k10_llmfilters` (no rerank) | 0.294 | 0.536 | 0.894 | 0.106 | 0.560 | 0.531 |
| `custom_parent_child_hybrid_k10_llmfilters_cross_encoderrerank_rt6` | 0.373 | 0.520 | 0.843 | 0.157 | 0.530 | 0.534 |
| `custom_parent_child_hybrid_a0.25_k10_llmfilters_cross_encoderrerank_rt6` ← **baseline for diversity experiments** | 0.346 | 0.565 | 0.871 | 0.129 | 0.520 | 0.554 |
| ↳ + `mmr_l0.4` (MMR lambda=0.4) | 0.399 | **0.633** | **0.933** | **0.067** | 0.550 | 0.562 |
| ↳ + `metaslots` (metadata_slots) | 0.358 | 0.549 | 0.926 | 0.074 | 0.540 | 0.567 |
| ↳ + `mmr_l0.4` + `pf` (per-filing) — first run after the Fix 1 prompt change | 0.357 | 0.628 | **0.938** | **0.062** | 0.560 | **0.582** |
| `custom_parent_child_hybrid_a0.25_k15_llmfilters_cross_encoderrerank_rt5` | 0.248 | 0.321 | 0.757 | 0.243 | 0.750 | 0.402 |
| `custom_semantic_hybrid_k10_llmfilters_cross_encoderrerank_rt6` (BGE-M3) | 0.333 | 0.453 | 0.800 | 0.200 | 0.580 | 0.533 |

> **Reading the table:** FER = factual_error_rate (lower is better, flagged when > 0.2).
> Hall. Rate = 1 − faithfulness. Ctx = context.

### Key observations from actual runs

1. **MMR (λ=0.4) gives the best recall and faithfulness** of all evaluated
   configurations: context_recall 0.633 (+6.8pp vs baseline), faithfulness
   0.933 (+6.2pp), hallucination 0.067 (−6.2pp). But FER slightly worsened
   to 0.55 — MMR pulls in temporally-adjacent chunks that the LLM over-reports
   as current-period findings (addressed by Fix 1 below).

2. **Metadata_slots gives better faithfulness than baseline** (0.926 vs 0.871)
   with lower variance than MMR, at the cost of slightly lower recall. It is
   more conservative than MMR — entity floors are met but globally-best chunks
   still fill remaining slots.

3. **Increasing k from 10 to 15 with rerank_top_k=5 hurt across the board**
   (FER 0.75, faithfulness 0.757) — the larger fetch pool included noisier
   candidates that the cross-encoder couldn't fully suppress at rt5.

4. **Semantic retrieval (no rerank, parent_child_semantic)** achieved the best
   context_precision (0.472) of any run, suggesting BGE-M3 embeddings match
   well for semantic-category questions even without reranking. FER still 0.50.

5. **FER floor problem:** Even the best run has FER=0.52–0.56 (13–14/25
   flagged). Root-cause breakdown: ~7 cases are retrieval failures (wrong
   chunks retrieved → high FER regardless of generation quality), ~4 are
   temporal over-reporting (Fix 1 target), ~10 are analytical questions where
   valid LLM interpretations diverge from the golden reference style.

6. **MMR + per-filing together give the best correctness and faithfulness
   of any recorded run** (correctness 0.582, faithfulness 0.938, hallucination
   0.062) — surpassing MMR alone and the semantic-retrieval run above on those
   two metrics. context_recall (0.628) and FER (0.56) land close to MMR-alone's
   numbers rather than improving further, so per-filing's marginal contribution
   on top of MMR looks small in this single run — and since this run also
   postdates the Fix 1 prompt change, some of the faithfulness/hallucination
   gain may belong to the prompt fix rather than retrieval. An isolated
   per-filing-only run is needed to separate the three effects.

### Key findings from earlier debugging

1. **Context precision was low (0.29)** because 1000-token parent chunks are
   full of boilerplate (legal disclaimers, table headers) that inflates both
   BM25 and vector similarity scores without being relevant.
2. **Context recall was low (0.55)** because `top_k=5` structurally caps how
   many sections can be retrieved for multi-section questions.
3. **Wrong-period retrieval** — early runs didn't pass company/year/quarter
   filters to the retriever, so a question about "Apple Q2 2023" could pull in
   Apple Q1 2023 or Q3 2022 chunks. Fixed by `src/retrieval/query_filters.py`.
4. **Alpha tuning (→ 0.25):** SEC questions hinge on exact tokens (tickers,
   dollar figures) that BM25 matches better than embeddings. alpha=0.25
   (BM25-heavy) improved recall on parent-child collections; had mixed results
   on the BGE-M3 semantic collection (different embedding model, different space).
5. **Generation prompt changes dominated faithfulness:** The rerank_top_k=6
   + prompt change in earlier runs pushed faithfulness from ~0.43 to 0.83 —
   reranking alone was not responsible; the generation rules ("REPRODUCE
   FIGURES EXACTLY", "[Source N]" citation format) drove the jump.

### Caveats — how to read this

- Latency is not measured. The evaluation pipeline has no timing
  instrumentation — add a timer around `_retrieve_chunks()` and `generate()`
  in `run_evaluation.py` if needed.
