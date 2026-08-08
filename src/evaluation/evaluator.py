"""
RAG evaluation orchestrator.

Computes ten metrics across three groups:

  ── Answer quality (require ground-truth reference) ─────────────────────────
  1. Exactness           src/evaluation/metrics.py  token F1, no LLM
  2. AnswerSimilarity    src/evaluation/metrics.py  embedding cosine, no LLM
  3. Correctness         RAGAS AnswerCorrectness    LLM + semantic (GEval-style)

  ── Retrieval quality (question + retrieved context + reference) ─────────────
  4. ContextPrecision    RAGAS ContextPrecision     useful chunks / total retrieved
  5. ContextRecall       RAGAS ContextRecall        golden claims covered by context
  6. ContextRelevance    custom GEval               holistic OpenAI rubric scoring

  ── Generation grounding (question + context + answer) ──────────────────────
  7. Faithfulness        RAGAS Faithfulness         answer claims grounded in context
  8. HallucinationRate   derived: 1 – faithfulness  per-question + overall %
  9. AnswerRelevance     RAGAS AnswerRelevancy       answer addresses the question
  10. FactualErrorRate   custom GEval               claims contradicted by golden

Usage::

    from src.evaluation.evaluator import EvalSample, evaluate_samples
    df = evaluate_samples(samples, config)
    print(df[["exactness", "answer_similarity", "correctness",
              "faithfulness", "context_relevance", "answer_relevance"]].mean())
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

import pandas as pd

from src.config import Config
from src.evaluation.metrics import (
    exactness,
    answer_similarity,
    geval_context_relevance,
    geval_factual_error_rate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric name constants
# ---------------------------------------------------------------------------

# Every metric name the evaluator can produce
ALL_METRICS: frozenset = frozenset({
    "exactness",          # token F1 vs golden answer         (rule-based)
    "answer_similarity",  # embedding cosine vs golden         (rule-based)
    "correctness",        # RAGAS AnswerCorrectness            (LLM)
    "context_precision",  # RAGAS ContextPrecision             (LLM)
    "context_recall",     # RAGAS ContextRecall                (LLM)
    "context_relevance",  # GEval holistic rubric              (LLM)
    "faithfulness",       # RAGAS Faithfulness                 (LLM)
    "hallucination_rate", # 1 – faithfulness (derived)
    "answer_relevance",   # RAGAS AnswerRelevancy              (LLM)
    "factual_error_rate", # GEval claim audit vs golden        (LLM)
})

# Subset of ALL_METRICS that are computed via RAGAS
_RAGAS_METRICS: frozenset = frozenset({
    "correctness", "faithfulness", "answer_relevance",
    "context_precision", "context_recall",
})


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    """One question-answer pair wired through the full RAG pipeline."""
    question: str
    golden_answer: str
    generated_answer: str
    retrieved_contexts: List[str] = field(default_factory=list)
    question_type: str = ""        # e.g. "Single-Doc Single-Chunk RAG", "Multi-Doc RAG"
    source_chunk_type: str = ""    # e.g. "Table", "Text"


# ---------------------------------------------------------------------------
# RAGAS helpers
# ---------------------------------------------------------------------------

def _build_ragas_dataset(samples: List[EvalSample]):
    """
    Convert EvalSample list → ragas EvaluationDataset (0.2+ API).

    Returns an ``EvaluationDataset`` ready to pass to ``ragas.evaluate()``.
    """
    from ragas import EvaluationDataset, SingleTurnSample  # type: ignore

    ragas_samples = [
        SingleTurnSample(
            user_input=s.question,
            retrieved_contexts=s.retrieved_contexts,
            response=s.generated_answer,
            reference=s.golden_answer,
        )
        for s in samples
    ]
    return EvaluationDataset(samples=ragas_samples)


def _run_ragas_metrics(
    samples: List[EvalSample],
    config: Config,
    requested: Set[str],
) -> pd.DataFrame:
    """
    Run only the *requested* RAGAS metrics (subset of _RAGAS_METRICS).

    Column names are normalised to snake_case.  Every column in _RAGAS_METRICS
    is present in the output; columns not in *requested* are NaN.
    """
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        AnswerCorrectness,
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from ragas.run_config import RunConfig
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=config.evaluation.model,
            api_key=config.evaluation.api_key,
            temperature=0,
        )
    )
    # AnswerCorrectness is a composite metric (factual LLM + semantic similarity).
    # The semantic-similarity component requires embeddings; without it the metric
    # silently returns NaN for every sample.
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=config.embedding.model,
            api_key=config.evaluation.api_key,
        )
    )

    _METRIC_CLASSES = {
        "correctness":       lambda: AnswerCorrectness(llm=llm, embeddings=embeddings),
        "faithfulness":      lambda: Faithfulness(llm=llm),
        "answer_relevance":  lambda: AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": lambda: ContextPrecision(llm=llm),
        "context_recall":    lambda: ContextRecall(llm=llm),
    }

    metrics_to_run = [
        _METRIC_CLASSES[m]()
        for m in requested
        if m in _METRIC_CLASSES
    ]

    dataset = _build_ragas_dataset(samples)

    logger.info(
        "Running RAGAS metrics %s on %d samples using judge model=%s ...",
        sorted(requested), len(samples), config.evaluation.model,
    )

    # Limit to 4 concurrent LLM calls to avoid OpenAI rate-limit errors that
    # cause per-sample NaN.  The default (max_workers=16) fires too many
    # simultaneous requests and the API silently drops them.
    run_cfg = RunConfig(max_workers=4, max_retries=3, timeout=180)

    result = ragas_evaluate(dataset=dataset, metrics=metrics_to_run, run_config=run_cfg)
    ragas_df = result.to_pandas()

    rename_map = {
        "answer_correctness": "correctness",
        "faithfulness":       "faithfulness",
        "answer_relevancy":   "answer_relevance",
        "answer_relevance":   "answer_relevance",
        "context_precision":  "context_precision",
        "context_recall":     "context_recall",
    }
    ragas_df = ragas_df.rename(columns=rename_map)

    for col in _RAGAS_METRICS:
        if col not in ragas_df.columns:
            ragas_df[col] = float("nan")

    return ragas_df[list(_RAGAS_METRICS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_samples(
    samples: List[EvalSample],
    config: Config,
    skip_ragas: bool = False,
    metrics: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Compute evaluation metrics for *samples* and return a DataFrame.

    Each row corresponds to one ``EvalSample``.  Columns always present
    (uncomputed ones are NaN)::

        question, golden_answer, generated_answer,
        exactness, answer_similarity,            # rule-based / embedding
        correctness,                             # RAGAS LLM
        faithfulness, context_relevance,         # RAGAS / custom GEval
        answer_relevance,                        # RAGAS LLM
        hallucination_rate,                      # derived: 1 – faithfulness
        factual_error_rate                       # GEval vs golden answer

    Args:
        samples:     List of evaluated samples (question + golden + generated + contexts).
        config:      Project-wide Config (used for embedding and generation settings).
        skip_ragas:  If True, skip RAGAS LLM metrics (useful for offline / CI runs).
        metrics:     Set of metric names to compute.  If None (default), all metrics
                     are computed.  Pass a subset of ALL_METRICS to skip the rest.
                     ``hallucination_rate`` automatically includes ``faithfulness``.

    Returns:
        pd.DataFrame with one row per sample and all metric columns.
    """
    if not samples:
        raise ValueError("evaluate_samples received an empty sample list.")

    n = len(samples)
    compute_all = metrics is None

    def want(m: str) -> bool:
        return compute_all or m in metrics  # type: ignore[operator]

    logger.info(
        "Starting evaluation of %d samples.%s",
        n,
        "" if compute_all else f"  Computing only: {sorted(metrics)}",  # type: ignore[arg-type]
    )

    # hallucination_rate is derived from faithfulness — include faithfulness in
    # the RAGAS run whenever either is requested.
    ragas_to_run: Set[str] = set()
    if not skip_ragas:
        ragas_to_run = {m for m in _RAGAS_METRICS if want(m)}
        if want("hallucination_rate"):
            ragas_to_run.add("faithfulness")

    # ── Step 1: Rule-based metrics (no LLM) ────────────────────────────────
    if want("exactness"):
        logger.info("Computing Exactness (token F1) ...")
        exact_scores = [exactness(s.generated_answer, s.golden_answer) for s in samples]
    else:
        exact_scores = [float("nan")] * n

    if want("answer_similarity"):
        logger.info("Computing AnswerSimilarity (embedding cosine) ...")
        sim_scores = [
            answer_similarity(s.generated_answer, s.golden_answer, config.embedding)
            for s in samples
        ]
    else:
        sim_scores = [float("nan")] * n

    # ── Step 2: GEval metrics (OpenAI rubric, per-question LLM calls) ───────
    if want("context_relevance"):
        logger.info("Computing ContextRelevance (GEval / OpenAI rubric) ...")
        ctx_scores = [
            geval_context_relevance(
                question=s.question,
                contexts=s.retrieved_contexts,
                api_key=config.evaluation.api_key,
                model=config.evaluation.model,
            )
            for s in samples
        ]
    else:
        ctx_scores = [float("nan")] * n

    if want("factual_error_rate"):
        logger.info("Computing FactualErrorRate (GEval / OpenAI claim-level audit) ...")
        factual_error_scores = [
            geval_factual_error_rate(
                question=s.question,
                generated=s.generated_answer,
                golden=s.golden_answer,
                api_key=config.evaluation.api_key,
                model=config.evaluation.model,
            )
            for s in samples
        ]
    else:
        factual_error_scores = [float("nan")] * n

    # ── Step 3: Assemble base DataFrame ────────────────────────────────────
    df = pd.DataFrame(
        {
            "question":           [s.question for s in samples],
            "question_type":      [s.question_type for s in samples],
            "source_chunk_type":  [s.source_chunk_type for s in samples],
            "golden_answer":      [s.golden_answer for s in samples],
            "generated_answer":   [s.generated_answer for s in samples],
            "exactness":          exact_scores,
            "answer_similarity":  sim_scores,
            "context_relevance":  ctx_scores,
            "factual_error_rate": factual_error_scores,
        }
    )

    # ── Step 4: RAGAS LLM metrics ──────────────────────────────────────────
    if skip_ragas and ragas_to_run:
        logger.warning("skip_ragas=True — RAGAS metrics will be NaN.")
        ragas_to_run = set()

    if ragas_to_run:
        try:
            ragas_df = _run_ragas_metrics(samples, config, ragas_to_run)
            for col in _RAGAS_METRICS:
                df[col] = ragas_df[col].values
        except Exception as exc:
            logger.error(
                "RAGAS evaluation failed (%s). Setting RAGAS metrics to NaN. "
                "Re-run with --skip-ragas to suppress this path.",
                exc,
            )
            for col in _RAGAS_METRICS:
                df[col] = float("nan")
    else:
        for col in _RAGAS_METRICS:
            df[col] = float("nan")

    # ── Step 5: Derive hallucination_rate from faithfulness ────────────────
    df["hallucination_rate"] = (
        (1.0 - df["faithfulness"]).clip(lower=0.0)
        if want("hallucination_rate")
        else float("nan")
    )

    col_order = [
        "question", "question_type", "source_chunk_type",
        "golden_answer", "generated_answer",
        "exactness", "answer_similarity", "correctness",
        "context_precision", "context_recall", "context_relevance",
        "faithfulness", "hallucination_rate", "answer_relevance",
        "factual_error_rate",
    ]
    return df[col_order]


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

METRIC_COLUMNS = [
    # Answer quality
    "exactness",
    "answer_similarity",
    "correctness",
    # Retrieval quality
    "context_precision",
    "context_recall",
    "context_relevance",
    # Generation grounding
    "faithfulness",
    "hallucination_rate",
    "answer_relevance",
    "factual_error_rate",
]

METRIC_DESCRIPTIONS = {
    "exactness":          "Token F1 overlap vs golden answer          (↑ better, no LLM)",
    "answer_similarity":  "Embedding cosine vs golden answer           (↑ better, no LLM)",
    "correctness":        "RAGAS AnswerCorrectness vs reference        (↑ better, LLM)",
    "context_precision":  "RAGAS ContextPrecision – useful chunks / retrieved  (↑ better, LLM)",
    "context_recall":     "RAGAS ContextRecall – golden claims in context      (↑ better, LLM)",
    "context_relevance":  "GEval holistic context relevance to question        (↑ better, LLM)",
    "faithfulness":       "RAGAS Faithfulness to retrieved ctx        (↑ better, LLM)",
    "hallucination_rate": "1 – faithfulness (context hallucination)   (↓ better, derived)",
    "answer_relevance":   "RAGAS AnswerRelevancy to question          (↑ better, LLM)",
    "factual_error_rate": "GEval factual error vs golden answer       (↓ better, LLM)",
}

# Columns where lower is better (used to flag direction in the summary)
_LOWER_IS_BETTER = {"hallucination_rate", "factual_error_rate"}


def build_hallucination_summary(df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """
    Build a small DataFrame capturing the hallucination summary block.

    Returns a DataFrame with columns::

        metric, avg_score, avg_pct, flagged_count, total, flagged_pct, threshold, note

    One row per hallucination-related metric (hallucination_rate, factual_error_rate).
    Intended to be appended to the aggregated summary CSV so it is persisted alongside
    the mean/std/min/max metrics.
    """
    n = len(df)
    rows = []

    for col, note in (
        ("hallucination_rate",  "1 – faithfulness (context-grounded); lower is better"),
        ("factual_error_rate",  "GEval claim audit vs golden answer; lower is better"),
    ):
        series = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
        if series.empty:
            rows.append({
                "metric": col, "avg_score": float("nan"), "avg_pct": float("nan"),
                "flagged_count": float("nan"), "total": n,
                "flagged_pct": float("nan"), "threshold": threshold, "note": note,
            })
        else:
            flagged = int((series > threshold).sum())
            rows.append({
                "metric": col,
                "avg_score": round(float(series.mean()), 4),
                "avg_pct": round(float(series.mean()) * 100, 2),
                "flagged_count": flagged,
                "total": n,
                "flagged_pct": round(flagged / n * 100, 2),
                "threshold": threshold,
                "note": note,
            })

    return pd.DataFrame(rows)


def _print_metric_table(df: pd.DataFrame, label: str) -> None:
    n = len(df)
    print()
    print("=" * 72)
    print(f"  {label}  ({n} samples)")
    print("=" * 72)
    print(f"  {'Metric':<24}  {'Dir':>3}  {'Mean':>6}  {'Std':>6}  {'Min':>6}  {'Max':>6}")
    print("-" * 72)
    for col in METRIC_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        direction = "↓" if col in _LOWER_IS_BETTER else "↑"
        if series.empty:
            print(f"  {col:<24}  {direction:>3}  {'N/A':>6}")
            continue
        print(
            f"  {col:<24}  {direction:>3}  {series.mean():6.4f}"
            f"  {series.std():6.4f}  {series.min():6.4f}  {series.max():6.4f}"
        )
    print("=" * 72)


def print_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary table of metric scores plus per-question-type breakdown."""
    _print_metric_table(df, "RAG EVALUATION SUMMARY")

    # Per question-type breakdown (only if the column is present and has variety)
    if "question_type" in df.columns:
        types = [t for t in df["question_type"].dropna().unique() if t]
        if len(types) > 1:
            for qtype in sorted(types):
                subset = df[df["question_type"] == qtype]
                _print_metric_table(subset, f"  ↳ {qtype}")

    n = len(df)
    # ── Hallucination summary block ────────────────────────────────────────
    print()
    print("  HALLUCINATION SUMMARY")
    print("  " + "-" * 50)

    ctx_hal = df["hallucination_rate"].dropna()
    if not ctx_hal.empty:
        pct = ctx_hal.mean() * 100
        flagged = (ctx_hal > 0.2).sum()
        print(f"  Context hallucination  (avg): {pct:5.1f}%  "
              f"({flagged}/{n} questions flagged > 20%)")
        print(f"  Per-question scores: {ctx_hal.round(2).tolist()}")

    fact_err = df["factual_error_rate"].dropna()
    if not fact_err.empty:
        pct = fact_err.mean() * 100
        flagged = (fact_err > 0.2).sum()
        print(f"  Factual error rate     (avg): {pct:5.1f}%  "
              f"({flagged}/{n} questions flagged > 20%)")
        print(f"  Per-question scores: {fact_err.round(2).tolist()}")

    print()
    print("  NOTE: hallucination_rate = 1 – faithfulness (context-grounded)")
    print("        factual_error_rate  = GEval claim audit vs golden answer")
    print("=" * 72)
    print()
