"""Pydantic request/response models for the RAG API."""
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    # Optional metadata filters – narrow retrieval to specific filings
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
            "When null, falls back to use_smart for compatibility."
        ),
    )
    use_smart: bool = Field(
        default=False,
        description=(
            "Legacy flag. True maps to chunking_strategy='parent_child'. "
            "Ignored when chunking_strategy is provided."
        ),
    )
    # Optional re-ranking after initial retrieval (None = disabled)
    rerank_mode: Optional[str] = Field(
        default=None,
        description=(
            "Re-rank over-fetched candidates before generation. "
            "'llm': one gpt-4o-mini call ranks all candidates. "
            "'cross_encoder': local sentence-transformers (no API cost). "
            "null/omit to skip re-ranking."
        ),
    )
    # Filter extraction mode when company/year/quarter are not passed explicitly
    filter_mode: str = Field(
        default="llm",
        description=(
            "'llm' (default): extract company/year/quarter from query text via LLM. "
            "'regex': rule-based extraction, no LLM cost. "
            "Ignored if company/year/quarter are provided explicitly."
        ),
    )
    # Per-request retrieval mode override (falls back to RETRIEVAL_MODE env var)
    retrieval_mode: Optional[str] = Field(
        default=None,
        description="Override retrieval mode: 'semantic' or 'hybrid'. Defaults to server config.",
    )
    retrieval_alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="BM25/vector balance for hybrid mode (0=BM25, 1=vector). Defaults to server config.",
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


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    query: str
    chunks_retrieved: int


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
