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
from typing import List, Optional

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
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    """One question-answer pair wired through the full RAG pipeline."""
    question: str
    golden_answer: str
    generated_answer: str
    retrieved_contexts: List[str] = field(default_factory=list)


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


def _run_ragas_metrics(samples: List[EvalSample], config: Config) -> pd.DataFrame:
    """
    Run RAGAS LLM-based metrics and return a DataFrame aligned with *samples*.

    Metrics:
      - AnswerCorrectness  → "correctness"
      - Faithfulness       → "faithfulness"
      - AnswerRelevancy    → "answer_relevance"

    Column names are normalised to snake_case to match our naming convention.
    Returns an empty DataFrame with NaN columns on any failure.
    """
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        AnswerCorrectness,
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=config.evaluation.model,
            api_key=config.evaluation.api_key,
            temperature=0,
        )
    )

    dataset = _build_ragas_dataset(samples)

    logger.info(
        "Running RAGAS metrics (AnswerCorrectness, Faithfulness, AnswerRelevancy, "
        "ContextPrecision, ContextRecall) on %d samples using judge model=%s ...",
        len(samples),
        config.evaluation.model,
    )

    result = ragas_evaluate(
        dataset=dataset,
        metrics=[
            AnswerCorrectness(llm=llm),
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm),
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
        ],
    )

    ragas_df = result.to_pandas()

    # Normalise column names (RAGAS may vary slightly across versions)
    rename_map = {
        "answer_correctness":  "correctness",
        "faithfulness":        "faithfulness",
        "answer_relevancy":    "answer_relevance",
        "answer_relevance":    "answer_relevance",   # legacy
        "context_precision":   "context_precision",
        "context_recall":      "context_recall",
    }
    ragas_df = ragas_df.rename(columns=rename_map)

    # Ensure all expected columns exist; fill missing ones with NaN
    for col in ("correctness", "faithfulness", "answer_relevance",
                "context_precision", "context_recall"):
        if col not in ragas_df.columns:
            ragas_df[col] = float("nan")

    return ragas_df[
        ["correctness", "faithfulness", "answer_relevance",
         "context_precision", "context_recall"]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_samples(
    samples: List[EvalSample],
    config: Config,
    skip_ragas: bool = False,
) -> pd.DataFrame:
    """
    Compute all eight evaluation metrics for *samples* and return a DataFrame.

    Each row corresponds to one ``EvalSample``.  Columns::

        question, golden_answer, generated_answer,
        exactness, answer_similarity,            # rule-based / embedding
        correctness,                             # RAGAS LLM
        faithfulness, context_relevance,         # RAGAS / custom GEval
        answer_relevance,                        # RAGAS LLM
        hallucination_rate,                      # derived: 1 - faithfulness
        factual_error_rate                       # GEval vs golden answer

    Args:
        samples:     List of evaluated samples (question + golden + generated + contexts).
        config:      Project-wide Config (used for embedding and generation settings).
        skip_ragas:  If True, skip RAGAS LLM metrics (useful for offline / CI runs).

    Returns:
        pd.DataFrame with one row per sample and all metric columns.
    """
    if not samples:
        raise ValueError("evaluate_samples received an empty sample list.")

    n = len(samples)
    logger.info("Starting evaluation of %d samples.", n)

    # ── Step 1: Rule-based metrics (no LLM) ────────────────────────────────
    logger.info("Computing Exactness (token F1) ...")
    exact_scores = [exactness(s.generated_answer, s.golden_answer) for s in samples]

    logger.info("Computing AnswerSimilarity (embedding cosine) ...")
    sim_scores = [
        answer_similarity(s.generated_answer, s.golden_answer, config.embedding)
        for s in samples
    ]

    # ── Step 2: GEval – Context Relevance (custom OpenAI rubric) ───────────
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

    # ── Step 2b: GEval – Factual Error Rate (hallucination vs golden) ───────
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

    # ── Step 3: Assemble base DataFrame ────────────────────────────────────
    df = pd.DataFrame(
        {
            "question": [s.question for s in samples],
            "golden_answer": [s.golden_answer for s in samples],
            "generated_answer": [s.generated_answer for s in samples],
            "exactness": exact_scores,
            "answer_similarity": sim_scores,
            "context_relevance": ctx_scores,
            "factual_error_rate": factual_error_scores,
        }
    )

    # ── Step 4: RAGAS LLM metrics ──────────────────────────────────────────
    _ragas_cols = ("correctness", "faithfulness", "answer_relevance",
                   "context_precision", "context_recall")

    if skip_ragas:
        logger.warning("skip_ragas=True – RAGAS metrics will be NaN.")
        for col in _ragas_cols:
            df[col] = float("nan")
    else:
        try:
            ragas_df = _run_ragas_metrics(samples, config)
            for col in _ragas_cols:
                df[col] = ragas_df[col].values
        except Exception as exc:
            logger.error(
                "RAGAS evaluation failed (%s). Setting RAGAS metrics to NaN. "
                "Re-run with --skip-ragas to suppress this path.",
                exc,
            )
            for col in _ragas_cols:
                df[col] = float("nan")

    # ── Step 5: Derive hallucination_rate from faithfulness ────────────────
    # hallucination_rate = fraction of answer claims NOT grounded in context.
    # When faithfulness is NaN (skip_ragas), this will also be NaN.
    df["hallucination_rate"] = (1.0 - df["faithfulness"]).clip(lower=0.0)

    # Reorder columns for readability
    col_order = [
        "question",
        "golden_answer",
        "generated_answer",
        # answer quality
        "exactness",
        "answer_similarity",
        "correctness",
        # retrieval quality
        "context_precision",
        "context_recall",
        "context_relevance",
        # generation grounding
        "faithfulness",
        "hallucination_rate",
        "answer_relevance",
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


def print_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary table of metric scores plus hallucination %."""
    n = len(df)
    print()
    print("=" * 72)
    print(f"  RAG EVALUATION SUMMARY  ({n} samples)")
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

    # ── Hallucination summary block ────────────────────────────────────────
    print()
    print("  HALLUCINATION SUMMARY")
    print("  " + "-" * 50)

    ctx_hal = df["hallucination_rate"].dropna()
    if not ctx_hal.empty:
        pct = ctx_hal.mean() * 100
        per_q = ctx_hal.apply(lambda x: "HALLUCINATED" if x > 0.2 else "ok")
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
