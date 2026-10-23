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

## 7b. Tool calling — calculators, then a fully agentic loop

**Problem addressed:** analytics-style gold questions (trend across quarters,
gross margin, effective tax rate, a segment's share of revenue) require
multi-step arithmetic over figures pulled from prose. The generation model
does that unreliably when it computes silently in its answer.

**Calculators (shared by both stages below):** `src/generation/tools.py` —
`FinancialAnalyticsTools` with `calculate_change`, `analyze_trend`,
`calculate_ratio`, `calculate_share_of_total`, `compare_across_entities`,
exposed as OpenAI function schemas. The math always runs in Python.

### Stage 1 — deterministic router + calculator (replaced in /chat and eval)

`src/generation/router.py` + `src/generation/agent.py`: one forced
`route` call classifies the question (`rag_only` / `tool_only` /
`rag_and_tool`) and picks a calculator; one forced call extracts that
calculator's arguments from the already-retrieved chunks; the result was
injected into `generate()` as a `[Calculation Result]` block (prompt rule 7).
Fixed 2–3 extra LLM calls, no follow-up if a figure was missing.

Measured (`custom_parent_child_hybrid_a0.25_k10_llmfilters_cross_encoderrerank_rt6`,
n=25, without → with `_tools`):

| metric | no tools | `_tools` |
|---|---|---|
| factual_error_rate | 0.55 | **0.25** |
| answer_relevance | 0.616 | **0.690** |
| correctness | **0.543** | 0.500 |
| exactness | **0.466** | 0.437 |
| faithfulness | **0.919** | 0.808 |
| context_recall | 0.588 | 0.582 |

Factual errors more than halved, but faithfulness fell — computed values
(percent changes, shares) don't appear verbatim in the retrieved contexts the
faithfulness judge sees, so they read as unsupported claims.

The standalone pipeline is still runnable via `scripts/run_agent.py`
(`answer_question()`); `route_and_calculate()` was removed with no callers left.

### Stage 2 — agentic tool-calling loop (current /chat and `--agentic`)

**Files:** `src/agentic/` (isolated package: `agent.py`, `tools.py`),
`AgentConfig` in `src/config.py`; integrated in `src/api/main.py` and
`scripts/run_evaluation.py`.

**Approach:** the model gets `search_filings` + the five calculators with
`tool_choice="auto"` and loops: it decides which searches to run (query and
company/year/quarter filters per search), which calculator to call with the
figures it found, and when to stop and answer. Calculator errors (e.g. a
missing figure) are returned to the model as `{"error": ...}` so it can search
again instead of silently dropping the calculation. Capped at
`AGENT_MAX_ITERATIONS` (default 6) tool turns, then one `tool_choice="none"`
turn forces an answer.

- **Retrieval is injected** (`search_fn`): /chat wraps its request's
  engine/chunking/mode/alpha/top_k/rerank settings; the eval CLI wraps
  `_retrieve_chunks()` with the run's settings, passing the agent's filters
  in place of `--filter-mode` extraction (per-filing, rerank, diversity all
  still apply per search). Explicit `company/year/quarter` on a /chat request
  are pinned and override the agent's filters.
- **Global `[Source N]` numbering** across searches; a chunk returned again
  keeps its first number, so citations stay stable. All retrieved chunks are
  returned as `sources` (/chat) and as `retrieved_contexts` (eval).
- **Same answer rules:** the system prompt is the generator's
  `_SYSTEM_PROMPT` plus an agent addendum — so answers are formatted and
  scored the same way as single-pass RAG.
- /chat responses now include `tool_calls` (trace) and `agent_iterations`.
- **/chat retrieval defaults** (`ChatConfig` in `src/config.py`, resolved
  once per request by `_with_server_defaults()` in `src/api/main.py`).
  Previously a request without `chunking_strategy` fell back to the *basic*
  `SecDocument` collection with top_k 5 and alpha 0.5, so agent searches
  silently ran on the weakest collection. Unset fields now default to
  `parent_child` / hybrid alpha 0.25 / top_k 10, overridable via
  `CHAT_CHUNKING_STRATEGY` / `CHAT_RETRIEVAL_ALPHA` / `CHAT_TOP_K` or per
  request. The cross-encoder model is now cached per process
  (`_load_cross_encoder` in `src/retrieval/reranker.py`) since the agent can
  rerank every search it makes.
  **Update (2026-10-23):** /chat now defaults to the best evaluation config
  apart from MMR. That's parent_child / hybrid (`CHAT_RETRIEVAL_MODE`) /
  alpha 0.25 / top_k 10, with 30 candidates cross-encoder re-ranked down to
  6 (`CHAT_RERANK_MODE=cross_encoder`, `CHAT_RERANK_TOP_K=6`). A request can
  turn re-ranking off with `rerank_mode: "none"`. MMR/diversity isn't
  available in /chat, so results can differ from the
  `…_rt6_mmr_l0.4` evaluation runs. The cross-encoder needs the server
  running under `rag_venv`; otherwise re-ranking is skipped with a warning.
- **`--agentic` evaluation runs default `--chunking-strategy` to
  `parent_child`** when neither it nor `--use-smart` is given
  (`resolve_chunking_strategy()` in `src/evaluation/run_tag.py`, shared by
  the CLI and the UI's run-tag lookup so both name the run the same). Alpha
  and top-k keep their normal evaluation defaults — only /chat defaults those.

**Trade-off:** open-ended cost/latency (typically 2–4 LLM turns plus one
retrieval per search, up to `max_iterations + 1` turns) vs. Stage 1's fixed
2–3 calls. Expect the same faithfulness caveat as Stage 1 for computed values.

**Status:** Implemented, evaluation pending. Run with `--agentic` (run tag
suffix `_agentic`; historical `_tools` files are Stage 1) or the React UI
"Agentic (tool calling)" checkbox.

### Stage 2b — planning in the loop + stronger first turn

**Problem addressed:** on broad / multi-period questions the agent ran a
couple of generic searches and stopped. Example: *"Outline the risk elements
associated with Microsoft's business and the evolution of these risks"* —
every search matched Item 1A Risk Factors, so it missed the MD&A
"Economic Conditions, Challenges, and Risks" section (the golden answer's
"dynamic and highly competitive industry" point is nearly verbatim from it)
and the Contingencies note (litigation; accrued legal liabilities
$264M → $181M → $177M → $597M across the four filings). It then said the
evolution "is not explicitly detailed", because nothing told it that
comparing filings is grounded rather than the inference rule 4 forbids.

**Approach (planning inside the loop, no extra LLM call):**

- Generic strategy rules in the agent addendum (`src/agentic/agent.py`),
  nothing question-specific: for broad / multi-part / multi-period questions,
  write a short plan (sub-topics × companies × periods) in the first reply
  and issue its searches in the same reply; phrase searches around the
  different 10-Q sections a subject appears in; for change-over-time
  questions, compare filings (constant / added / dropped / figures moved),
  explicitly allowed as grounded synthesis; check the plan's coverage before
  answering. Single-fact questions skip the plan.
- **First turn on a stronger model** — `AGENT_PLANNER_MODEL` (default
  `gpt-4o`) runs turn 1 only, where the plan and opening searches are
  decided; later turns and the answer stay on `AGENT_MODEL` (`gpt-4o-mini`).
  The plan text is logged and returned as `agent_plan` by /chat.
- **Judge moved to `gpt-4.1`** (`EVALUATION_MODEL`) so the judge stays
  stronger than every answering model (gpt-4o-mini, gpt-4o planner). GPT-5 /
  o-series were ruled out: the pinned ragas 0.2 / langchain-openai 0.1
  stack sends `temperature=0` / `max_tokens`, which reasoning models reject.
  Runs judged by anything other than gpt-4o get a `_judge-<model>` run-tag
  suffix and a `_meta_judge_model` summary row, since scores from different
  judges aren't comparable; the CLI warns if the judge is also one of the
  run's answering models (self-preference bias).

**Observed (live, 2026-10-16):** asking for the plan in the prompt didn't
work — gpt-4o issued tool calls without any plan text, and all searches
stayed on "risk factors". Replaced by Stage 2c below.

### Stage 2c — forced `plan_research` tool

**Files:** `src/agentic/tools.py` (`plan_research` tool, `FILING_PARTS`),
`src/agentic/agent.py` (turn 1 forced via `tool_choice`).

**Approach:** turn 1 (on `AGENT_PLANNER_MODEL`) is forced by the API to
call `plan_research`, which takes a structured list of sub-topics, each with
`filing_part` (financial statements / notes / MD&A / market risk / legal
proceedings / risk factors / any), `search_query`, `companies` and
`periods`, plus `compare_across_periods`. The plan is recorded, returned
as `agent_plan` by /chat, and stays in the conversation as a checklist.
Costs one extra LLM turn per question (`AGENT_MAX_ITERATIONS` default raised
6 → 7 to compensate).

`filing_part` steers how the query is worded and is **not** a metadata
filter. Ingested `section_title` values are too unreliable to filter on:
AAPL's largest bucket is "Item 6. Exhibits" (100 chunks) and 73 more sit
under a sentence fragment; INTC has 149 under "Note 13 :" and 127 under
"PREAMBLE". A hard filter would silently drop relevant passages. Fixing
`section_title` at ingestion is a prerequisite for a real section filter.

**Observed (live /chat runs, parent_child, alpha 0.25, top_k 10; not a
formal evaluation):**

| Question | Result |
|---|---|
| Microsoft risks + evolution | Plan covered risk factors, MD&A, market risk and contingencies (12 parallel searches, 3 turns). The answer now includes litigation. But the plan skipped the 2022 Q3 filing, the queries stayed terse, the legal-accrual figures didn't make it in, and the "evolution" bullets are generic. 90 sources reached the context. |
| Apple net sales over time | All four figures correct ($82,959M → $117,154M → $94,836M → $81,797M) and the up-then-down pattern matches the golden answer, so the earlier $123,945M error didn't recur. But the plan invented periods that don't exist (2022 Q1/Q2/Q4, 2023 Q4), `analyze_trend` failed on a null value, and the run hit the turn cap (8 turns). |
| Apple net sales, quarter ended July 1, 2023 | **Wrong**: the plan mapped "July 1, 2023" to the 2023 Q2 filing and answered $94,836M; the correct figure is $81,797M (2023 Q3 filing). |

**Open issue (fixed in Stage 2d):** the agent didn't know which filings
existed or which dates each filing label covered, so it guessed periods,
wasting searches and sometimes picking the wrong filing.

### Stage 2d — fiscal-period dates at ingestion + filing catalog

**Problem addressed:** filing labels are not calendar quarters and differ
by company — the quarter ended July 1, 2023 is "2023 Q3" for AAPL but
"2023 Q2" for INTC; AAPL/MSFT "2023 Q1" ends Dec 31, 2022 while AMZN's ends
Mar 31, 2023. No reasoning from a date alone maps it to the right label.

**Ingestion** (`src/ingestion/filing_period.py`, called from `parse_pdf`):
- `period_end` comes from the cover line "For the quarterly period ended
  …", which every 10-Q must carry. It was found in all 20 filings.
  Statement headers aren't used for it, because their first date can be a
  prior-year column.
- `period_start` is derived. Calendar-quarter filers, detected because the
  statements compare against the same day a year earlier, get the calendar
  quarter. 52/53-week filers, which are also the default, get end minus 13
  weeks. All 20 filings classified correctly with no overlapping ranges.
  Known approximation: Apple's 14-week quarter ended Dec 31, 2022 is treated
  as 13 weeks, so it starts Oct 2 rather than Sep 25.
- Stored as Weaviate `DATE` properties `period_start`/`period_end` on
  `SecDocument`, `SecDocumentSmart` and `DocumentChunk`, and as YYYYMMDD
  ints in the LlamaIndex collections (excluded from embedded/LLM text).
  Dates come from the freshly parsed PDFs, not the chunk checkpoints, so
  re-ingesting from existing checkpoints adds them without re-embedding.
  `ensure_schema*` adds the properties to existing collections.

**Retrieval** (`src/retrieval/retriever.py`): `retrieve(period_date=…)`
filters `period_start <= date <= period_end`; chunks carry
`period_start`/`period_end`; `list_filing_periods()` reads the filing
catalog live from the index (one aggregate plus one query per filing), so
nothing needs rebuilding after new ingestion.

**Agent:**
- `search_filings` takes either the filing label (`year` + `quarter`) or
  `date`. The date wins if both are given.
- The filing catalog (label → date range) is placed in the first user
  message. The system prompt stays static, so it remains cacheable.
- Instructions: match question periods to catalog filings (labels for
  "Q2 2023" / "the Q3 2022 10-Q", dates for "quarter ended …" / "around
  May 2023"), plan only filings that exist, and check that column dates
  match the period asked about.
- Source headers now show "quarter ended …".
- /chat caches the catalog for 5 minutes; the evaluation CLI loads it once
  per `--agentic` run.

Labels were kept rather than replaced with dates because 80 of the 195 gold
questions name the filing label ("Apple's Q1 2023 10-Q"). Converting that
to a calendar date would pick the wrong Apple filing.

**Observed (live agent, OpenAI models, stand-in keyword search with the same
filters — Weaviate was down; not a formal evaluation):**

| Question | Before | After |
|---|---|---|
| Apple net sales, quarter ended Jul 1, 2023 | ❌ $94,836M (2023 Q2 filing) | ✅ $81,797M (2023 Q3) |
| Intel net revenue, quarter ended Jul 1, 2023 | — | ✅ $12,949M (2023 Q2, the same date maps to a different label) |
| Apple net sales over time | figures right, but searched 4 nonexistent periods; hit the turn cap (8 turns) | ✅ only the 4 real filings, `analyze_trend` +41.22% / −19.05% / −13.75%, overall −1.4%, "mixed"; 4 turns |
| Apple opex vs revenue, Q1 2023 (label-style gold question) | — | ✅ 2023 Q1 filing: $14,316M vs $117,154M; but the 12.2% ratio was computed by the model, not `calculate_ratio` |

In these runs the planner resolved dates through the catalog and searched
by label, so the `date` filter itself (unit-tested) wasn't exercised by the
model.

**Requires re-ingestion** of each collection you query (see README / Makefile
`ingest` targets). Until then, the catalog shows "period unknown" and date
filters on a collection without the properties fail.

### Stage 2e — collapse repeated passages + topic inventory before answering

**Problem addressed:** on broad multi-period questions the answer dropped
topics that *were* retrieved. Microsoft risk question, 3 runs on identical
retrieval: tax and litigation passages were in the sources every time, but
reached the answer in 1/3 runs (tax) and 0/3 (litigation). Causes:
- 10-Q text repeats each quarter, so about 40 retrieved passages were
  roughly 10 distinct ones ×4;
- one answer section per period exhausted the output on the same risks;
- nothing checked that the answer covered what the sources contained.

**Approach (`src/agentic/tools.py`, `src/agentic/agent.py`):**
1. **Repeat collapsing.** A passage is reported as "Repeats [Source N]"
   instead of in full when ≥80% of its 5-word shingles already appeared in
   sources from another filing of the same company AND all its figures (numbers with
   a comma, decimal or 4+ digits — bare small integers are page numbers)
   appear there too. Measured first: about half of MSFT 2023 Q1 parent chunks are
   ≥85% contained in 2022 Q3; income-statement chunks overlap 8–60% (never
   collapsed). Collapsed passages aren't numbered, so `sources` only contains
   distinct passages.
2. **Topic inventory (`list_topics`).** For questions the plan marks as
   multi-period or multi-part, the search turns end with "READY", and then
   `list_topics` is forced on `AGENT_PLANNER_MODEL` (gpt-4o). It lists one
   entry per specific subject in the sources, each with its sources, filings
   and change over time. **Every source must be accounted for** (under a topic
   or in `not_relevant`); otherwise the tool returns an error naming them
   and the step is retried once. The answer is then forced from the list.
   Exposed as `agent_topics` on /chat.

**Observed (MSFT risk question, 5 runs each, live Weaviate):**

| | Before | Repeats + topic list (gpt-4o-mini) | + gpt-4o list, accounting |
|---|---|---|---|
| distinct sources / repeat refs | 37 / 4 | 26 / 23 (page numbers fixed, 0.80 threshold) | 26 / 23 |
| topic list | — | "risk factors" (repeated the plan) | 5 specific topics, same every run |
| golden topics in answer (of 6) | 3–5, unstable | 3–4 | 3, deterministic |
| tax in answer | 1/3 | 5/5 | 5/5 |
| litigation in answer | 0/3 | 0/5 | 0/5 |

The source accounting showed why the coverage ceiling is 3: gpt-4o
**explicitly marks the MD&A "Economic Conditions, Challenges, and Risks"
passages and "Legal and Other Contingencies" as not relevant**, in every
run. Those passages hold golden points 1, 2, 4 and 5 almost verbatim. The
plan's single sub-topic ("risk factors", filing part = risk factors)
narrows what counts as relevant, and the inventory follows it. The next
lever is planner breadth, deliberately deferred until the evaluation below.

**Status:** Implemented. Multi-Doc evaluation (25 questions, seed 42,
`--agentic --alpha 0.25 --top-k 10`, judge gpt-4.1; files
`evaluation_results/evaluation_results_{pre_dedup,dedup_topics}_custom_parent_child_hybrid_a0.25_k10_llmfilters_multidoc_agentic_judge-gpt-4.1*.csv`):

| metric | before | after | Δ |
|---|---|---|---|
| correctness | 0.540 | 0.561 | +0.021 |
| answer_relevance | 0.800 | 0.838 | +0.038 |
| exactness | 0.407 | 0.423 | +0.015 |
| context_precision | 0.207 | 0.231 | +0.024 |
| faithfulness | 0.925 | 0.845 | **−0.080** |
| hallucination_rate | 0.075 | 0.155 | **+0.080** |
| context_recall | 0.813 | 0.754 | −0.059 |
| factual_error_rate | 0.130 | 0.160 | +0.030 |

Per question, correctness improved on 7, got worse on 5 and stayed the same
on 13. Gains: Apple buybacks +0.32, Services trend +0.24, net sales trend
+0.16. Losses: Amazon risk factors −0.21, Microsoft R&D vs prior quarters
−0.21.

**Faithfulness regression, likely cause:** answers claiming a topic was
"unchanged / consistently reported" doubled (4 → 8) and lost the most
faithfulness (−0.137 average vs −0.053). Collapsed repeats aren't kept as
sources, so the text from later filings that backs "unchanged in 2023 Q3"
isn't in the evaluation contexts (or the /chat sources). The judge can't
verify the claim, and context_recall drops for the same reason. Single
run, n=25, so part of the differences is noise.

**Fix (implemented):** repeats stay out of the prompt but are returned as
`AgentRunResult.repeated_sources` (chunk + the [Source N] numbers it
repeats). /chat returns them as `repeated_sources` (`repeats_sources` on
each entry), and `--agentic` evaluation adds them to the judged contexts.
`sources` numbering is unchanged. Tested offline. **Re-evaluation pending:
the OpenAI account ran out of credits (`insufficient_quota`) before it
could run.**

**Environment caveat:** both runs above were executed with the Anaconda
base Python (ragas 0.2.15, openai 1.30), not the project's `rag_venv`
(ragas 0.2.6, openai 1.51). They're comparable with each other but not
with `rag_venv` runs, and neither used reranking. The base environment's
torch is broken, so `--rerank-mode cross_encoder` there silently skipped
reranking and diversity. `src/retrieval/reranker.py` now logs the real
import error and says reranking and diversity were skipped. Re-run
everything in `rag_venv`.

### Agentic vs single-pass on the best retrieval config (rag_venv, gpt-4.1 judge)

Setup: `qna_data_gold_page_present_25.csv` (25 questions: 9 single-chunk,
8 multi-chunk, 8 Multi-Doc), parent_child / hybrid / alpha 0.25 / top-k 10
→ cross-encoder → rt6 → MMR λ 0.4. Run in `rag_venv`, no failures and no
skipped reranking. The agentic run is the current code (plan, dates,
repeat collapsing, topic list, repeats returned as contexts). Files:
`evaluation_results_custom_parent_child_hybrid_a0.25_k10_llmfilters_cross_encoderrerank_rt6_mmr_l0.4{_judge-gpt-4.1,_agentic_judge-gpt-4.1}.csv`.

| metric | original (gpt-4o judge) | single-pass (gpt-4.1) | agentic (gpt-4.1) | agentic − single |
|---|---|---|---|---|
| correctness | 0.562 | 0.589 | 0.550 | −0.039 |
| answer_similarity | 0.843 | 0.845 | 0.871 | +0.026 |
| faithfulness | 0.933 | 0.929 | 0.815 | −0.114 |
| factual_error_rate | 0.550 | 0.220 | 0.060 | −0.160 |
| answer_relevance | 0.534 | 0.469 | 0.902 | +0.433 |
| context_precision | 0.399 | 0.469 | 0.515 | +0.047 |
| context_recall | 0.633 | 0.580 | 0.775 | +0.195 |
| context_relevance | 0.540 | 0.700 | 0.790 | +0.090 |

Correctness by question type, single-pass → agentic:
- Multi-Doc: 0.515 → 0.591 (but faithfulness 0.93 → 0.75)
- Single-Doc Multi-Chunk: 0.527 → 0.445
- Single-Doc Single-Chunk: 0.710 → 0.607

Per question, the agent was better on 9, worse on 12 and about the same on
4. Re-judging the same single-pass answers with gpt-4.1 moved
factual_error_rate from 0.55 to 0.22, confirming that results from
different judges can't be compared.

**Why single-document questions lost (4 worst):**
1. The agent answers unanswerable questions with substitute figures where
   the golden answer says "not provided" (NVIDIA revenue vs costs, Intel
   new-product revenue vs marketing).
2. Its answers include more than was asked (NVIDIA R&D: added share of
   revenue and the prior-year comparison), which adds unsupported or extra
   statements.
3. It names periods by filing label ("Q3 2023") instead of the end date the
   golden answer uses (iPhone revenue: correct figure, but 0.60 vs 0.97).
4. "Latest" is ambiguous: the agent picked the newest filing in the
   catalog, while the golden answer used an earlier one.

**Generic prompt fixes (implemented, evaluation pending):** two new rules in
the agent instructions (`src/agentic/agent.py`):
- **ANSWER SCOPE:** lead with the direct answer, add nothing that wasn't
  asked, and when the filings lack what's asked, say so first and only
  then note related information; never build a substitute answer.
- **PERIOD NAMING:** refer to periods by quarter-end date ("the quarter
  ended July 1, 2023"), with the filing label optional in parentheses.

Spot check (not an evaluation):
- iPhone question: now "For the quarter ended July 1, 2023 (AAPL 2023
  Q3) … $39,669 million". Fixed.
- Intel unanswerable question: now says "not explicitly detailed" up
  front, but still goes on to compare total revenue with MG&A and refers
  to "Q3 2023" in places. Only partly followed by gpt-4o-mini, which writes
  the answer.

**Why only partly — instruction conflicts, not mainly the model.** Trace of
the Intel run:
1. The topic list had no way to say "not reported", so gpt-4o filed new-
   product revenue under sources 1–17. The "cover every topic" instruction
   then forced the answer to say something about it.
2. Generator rule 4's template ("…does not include X. Based on available
   sources: …") invites a substitute answer.
3. Rule 6 ("list each figure, then compare") pushed a comparison of
   substitutes.
4. "Q3 2023" was copied from Intel's own MD&A wording and the topic notes.

**A/B (2 questions, best retrieval config, gpt-4.1 judge, 1 run each;
prompt fix applied in memory):**

| variant | iPhone correctness | Intel correctness | Intel faithfulness |
|---|---|---|---|
| A gpt-4o-mini, current prompt | 0.97 | 0.31 | 0.80 |
| B gpt-4o, current prompt | 0.97 | 0.37 | 1.00 |
| C gpt-4o-mini, fixed prompt | 0.97 | 0.48 | 1.00 |
| D gpt-4o, fixed prompt | 0.97 | 0.58 | 1.00 |

The prompt fix is the larger lever (+0.17), the model adds more (+0.10),
and D matches single-pass (0.60). answer_relevance is 0 for every Intel
answer: the metric penalises "not reported" answers, including golden-like
ones.

**Adopted (D):**
- `list_topics` entries carry `reported` (bool). Sources are listed only if
  they actually report the topic, and reported=false topics must be stated
  as not reported, with no substitute figures.
- New agent rule "MISSING INFORMATION OVERRIDES RULES 4 AND 6": rule 4's
  continuation may only give asked-for information, and a comparison with
  a missing side is declared impossible. Period naming now also overrides
  labels taken from the sources and notes.
- `AGENT_MODEL` default gpt-4o-mini → **gpt-4o** (search turns + answer;
  the planner and topic steps were already gpt-4o). The judge stays gpt-4.1.
- Higher cost per question than gpt-4o-mini.

Live check after the change: the Intel answer reproduces D (topic marked
reported=false, "a direct comparison … cannot be made", periods as dates).
Minor quirk: it cited a source range ("[Sources 1-5, 7-9, 11-17]").
**Full evaluation pending.**

**Status:** Implemented, evaluation pending. **Re-run the non-agentic
baseline with the gpt-4.1 judge too** — every result above was judged by
gpt-4o, so agentic `_judge-gpt-4.1` numbers can't be compared against them
directly.

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
