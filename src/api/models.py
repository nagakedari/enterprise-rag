"""Pydantic request/response models for the RAG API."""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question to answer")
    # chunking_strategy / retrieval_alpha / top_k left null fall back to
    # server defaults (src/config.py ChatConfig: parent_child / 0.25 / 10).
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description=(
            "Chunks to retrieve per agent search; with rerank enabled, 3x this "
            "many candidates are fetched and reranked down to rerank_top_k. "
            "Defaults to CHAT_TOP_K (10)."
        ),
    )
    # Optional metadata filters – pin every agent search to specific filings
    # (otherwise the agent picks filters per search itself)
    company: Optional[str] = Field(default=None, description="Ticker symbol filter, e.g. 'AAPL'")
    year: Optional[int] = Field(default=None, description="Filing year filter, e.g. 2023")
    quarter: Optional[str] = Field(default=None, description="Quarter filter, e.g. 'Q2'")
    # Retrieval engine and chunking strategy
    engine: str = Field(
        default="custom",
        description="Retrieval engine: 'custom' (direct Weaviate client) or 'llamaindex'",
    )
    chunking_strategy: Optional[str] = Field(
        default=None,
        description=(
            "Which ingested collection to query: "
            "'basic' → SecDocument, "
            "'parent_child' → SecDocumentSmart, "
            "'semantic' → DocumentChunk (BGE-M3). "
            "When null: 'parent_child' if use_smart, else CHAT_CHUNKING_STRATEGY "
            "(default 'parent_child')."
        ),
    )
    use_smart: bool = Field(
        default=False,
        description=(
            "Legacy flag. True maps to chunking_strategy='parent_child'. "
            "Ignored when chunking_strategy is provided."
        ),
    )
    # Re-ranking after initial retrieval
    rerank_mode: Optional[str] = Field(
        default=None,
        description=(
            "Re-rank over-fetched candidates before they reach the agent. "
            "'llm': one gpt-4o-mini call ranks all candidates. "
            "'cross_encoder': local sentence-transformers (no API cost). "
            "'none': no re-ranking. "
            "null/omit: CHAT_RERANK_MODE (default 'cross_encoder')."
        ),
    )
    rerank_top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description=(
            "Chunks kept per search after re-ranking. Ignored when re-ranking "
            "is disabled. Defaults to CHAT_RERANK_TOP_K (6)."
        ),
    )
    # Kept for request compatibility only — /chat is agentic and the agent
    # chooses company/year/quarter filters per search itself.
    filter_mode: str = Field(
        default="llm",
        description=(
            "Unused by /chat: the agent chooses company/year/quarter filters "
            "for each search itself. Accepted for backward compatibility."
        ),
    )
    # Per-request retrieval mode override (falls back to RETRIEVAL_MODE env var)
    retrieval_mode: Optional[str] = Field(
        default=None,
        description="Override retrieval mode: 'semantic' or 'hybrid'. Defaults to CHAT_RETRIEVAL_MODE ('hybrid').",
    )
    retrieval_alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "BM25/vector balance for hybrid mode (0=BM25, 1=vector). "
            "Defaults to CHAT_RETRIEVAL_ALPHA (0.25)."
        ),
    )


class SourceDocument(BaseModel):
    company: str
    quarter: str
    year: int
    source_file: str
    chunk_index: int
    page_start: int
    page_end: int
    score: float = Field(description="Similarity score in [0, 1], higher = more relevant")
    text_preview: str = Field(description="First 200 characters of the chunk")
    repeats_sources: Optional[List[int]] = Field(
        default=None,
        description="Set on repeated_sources entries: the [Source N] numbers this passage repeats.",
    )


class ToolCallTrace(BaseModel):
    """One tool call the agent made while answering (src/agentic/tools.py)."""
    name: str
    arguments: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument] = Field(
        description="Every chunk the agent retrieved; sources[i] is [Source i+1] in the answer.",
    )
    query: str
    chunks_retrieved: int
    tool_calls: List[ToolCallTrace] = Field(
        default_factory=list,
        description="Searches and calculations the agent ran, in order.",
    )
    repeated_sources: List[SourceDocument] = Field(
        default_factory=list,
        description=(
            "Passages from other filings that repeat a numbered source's text "
            "(collapsed in the agent's prompt). They back claims that something "
            "was reported unchanged in those filings."
        ),
    )
    agent_iterations: int = Field(default=0, description="LLM turns the agent loop took.")
    agent_plan: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured plan (plan_research arguments) the agent made in its first turn.",
    )
    agent_topics: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Topic inventory (list_topics) the answer was written from; broad questions only.",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation models — mirror scripts/run_evaluation.py's CLI flags 1:1.
#
# Fields left as None/Optional represent an OMITTED CLI flag, not "use the
# code default value" — the run-tag algorithm (src/evaluation/run_tag.py)
# only appends a tag segment when the corresponding field is not None, so
# the frontend must serialize an unset field as null, never as a filled-in
# default number, or generated run tags will drift from historical filenames.
# ═════════════════════════════════════════════════════════════════════════════

class EvaluationRunParams(BaseModel):
    engine: Literal["custom", "llamaindex"] = "custom"
    chunking_strategy: Optional[Literal["basic", "parent_child", "semantic"]] = None
    use_smart: bool = Field(
        default=False,
        description="Legacy flag: True maps to chunking_strategy='parent_child'. Ignored when chunking_strategy is set.",
    )
    retrieval_mode: Optional[Literal["semantic", "hybrid"]] = None
    alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    filter_mode: Literal["llm", "regex"] = "llm"
    rerank_mode: Optional[Literal["llm", "cross_encoder"]] = None
    rerank_top_k: Optional[int] = Field(default=None, ge=1, le=50)
    diversity_mode: Literal["none", "mmr", "metadata_slots", "source_cap"] = Field(
        default="none",
        description=(
            "Final chunk selection strategy after reranker scoring — replaces "
            "plain top-k sort. 'mmr': Maximal Marginal Relevance (requires "
            "scikit-learn). 'metadata_slots': guaranteed per-(company, quarter, "
            "year) slot coverage. 'source_cap': per-entity upper-bound cap. "
            "'none' (default): sort by score."
        ),
    )
    mmr_lambda: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MMR relevance/diversity trade-off. 1.0=pure relevance, 0.0=pure diversity.",
    )
    per_filing: bool = Field(
        default=False,
        description=(
            "Run one retrieval query per (year, quarter) filing when the question "
            "is temporal (no quarter filter). Guarantees facts from every filing "
            "period are in the candidate pool. Ignored when quarter is specified or "
            "engine='llamaindex'."
        ),
    )
    chunks_per_filing: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Chunks to retrieve per (year, quarter) filing when per_filing=True.",
    )
    max_per_entity: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description=(
            "Override for source_cap per-(company, quarter, year) slot cap. "
            "Only used when diversity_mode='source_cap'. "
            "Defaults to ceil(top_k / n_entities), minimum 2."
        ),
    )
    agentic: bool = Field(
        default=False,
        description=(
            "Answer with the agentic tool-calling loop (src/agentic/agent.py), "
            "same as /chat: the model decides which searches to run (through "
            "this run's retrieval settings) and which FinancialAnalyticsTools "
            "calculator to call, instead of one retrieval + one generation. "
            "Default off, so existing evaluation runs are unaffected."
        ),
    )
    samples: int = Field(default=25, ge=1, le=500)
    company: Optional[str] = None
    question_type: Optional[str] = None
    metrics: Optional[List[str]] = Field(
        default=None,
        description="Subset of ALL_METRICS to compute. None/omitted = compute every metric.",
    )
    skip_ragas: bool = False
    seed: int = 42
    input_path: Optional[str] = Field(
        default=None,
        description="Advanced override for --input (path to a golden Q&A CSV). None = use the CLI's own default.",
    )

    @field_validator("metrics")
    @classmethod
    def _validate_metrics(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            from src.evaluation.evaluator import ALL_METRICS
            unknown = set(v) - ALL_METRICS
            if unknown:
                raise ValueError(f"Unknown metric name(s): {sorted(unknown)}")
        return v


class StartRunRequest(EvaluationRunParams):
    force: bool = Field(
        default=False,
        description="Bypass the 409 conflict guard and launch even if an exact-match result already exists.",
    )


class HistoryRunSummary(BaseModel):
    id: str = Field(description="Opaque server-derived id — never a client-supplied path.")
    source: Literal["root", "evaluation_results_dir"]
    csv_path: str = Field(description="Per-sample results CSV path, relative to the repo root.")
    summary_csv_path: Optional[str] = None
    display_name: str
    run_tag_guess: Optional[str] = Field(
        default=None, description="Best-effort parsed run tag; None for unparseable legacy filenames."
    )
    parsed_params: dict = Field(default_factory=dict)
    modified_at: datetime
    row_count: Optional[int] = None


class LookupResponse(BaseModel):
    requested_run_tag: str
    exact_match: Optional[HistoryRunSummary] = None
    close_matches: List[HistoryRunSummary] = Field(default_factory=list)


class MetricRow(BaseModel):
    metric: str
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    avg_pct: Optional[float] = None
    flagged_count: Optional[float] = None
    total: Optional[float] = None
    flagged_pct: Optional[float] = None
    threshold: Optional[float] = None
    note: Optional[str] = None


class SampleRow(BaseModel):
    question: str
    question_type: Optional[str] = None
    source_chunk_type: Optional[str] = None
    golden_answer: str
    generated_answer: str
    exactness: Optional[float] = None
    answer_similarity: Optional[float] = None
    correctness: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    context_relevance: Optional[float] = None
    faithfulness: Optional[float] = None
    hallucination_rate: Optional[float] = None
    answer_relevance: Optional[float] = None
    factual_error_rate: Optional[float] = None


class RunDetail(BaseModel):
    summary: HistoryRunSummary
    metrics: List[MetricRow]
    samples: List[SampleRow]
    meta: dict = Field(
        default_factory=dict,
        description="Self-documenting run metadata (input_file, run_tag, n_questions), when present.",
    )


class OptionsResponse(BaseModel):
    engines: List[str]
    chunking_strategies: List[str]
    retrieval_modes: List[str]
    filter_modes: List[str]
    rerank_modes: List[str]
    diversity_modes: List[str]
    metrics: List[str]
    default_top_k: int
    default_samples: int
    default_seed: int


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    run_tag: str
    params: EvaluationRunParams
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress_current: Optional[int] = None
    progress_total: Optional[int] = None
    log_tail: List[str] = Field(default_factory=list)
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    output_csv_path: Optional[str] = None
    summary_csv_path: Optional[str] = None


class StartRunResponse(BaseModel):
    job_id: str
    run_tag: str
    status: str
