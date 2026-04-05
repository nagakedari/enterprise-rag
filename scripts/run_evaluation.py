"""
RAG evaluation CLI.

Loads golden Q&A pairs from the SEC 10-Q dataset, runs the full RAG pipeline
for each question, then computes six evaluation metrics and saves the results.

Usage examples
--------------
# Evaluate 25 random questions (default) and save to evaluation_results.csv
python scripts/run_evaluation.py

# Evaluate 50 questions and save to a custom path
python scripts/run_evaluation.py --samples 50 --output results/eval_run1.csv

# Run only rule-based metrics (no RAGAS LLM calls, fast / cheap)
python scripts/run_evaluation.py --skip-ragas

# Filter to a specific company
python scripts/run_evaluation.py --company AAPL --samples 20

Metrics
-------
  Golden-answer metrics (require reference):
    exactness           Token F1 overlap vs golden answer
    answer_similarity   Embedding cosine similarity vs golden answer
    correctness         RAGAS AnswerCorrectness (LLM + semantic)

  RAG-quality metrics (GEval-style LLM scoring):
    faithfulness        RAGAS – is the answer grounded in the retrieved context?
    context_relevance   Custom GEval – are the contexts relevant to the question?
    answer_relevance    RAGAS – is the answer relevant to the question?
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ── Project root on sys.path ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from src.config import Config
from src.retrieval.retriever import retrieve
from src.generation.generator import generate
from src.evaluation.evaluator import (
    EvalSample,
    evaluate_samples,
    print_summary,
    METRIC_COLUMNS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QNA_CSV = Path(
    "/Users/manjusri/learning/generative_ai/KG-RAG-datasets/sec-10-q/data/v1/qna_data.csv"
)
DEFAULT_OUTPUT = Path("evaluation_results.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_qna(
    csv_path: Path,
    n_samples: int,
    company_filter: str | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Load Q&A pairs from *csv_path*, optionally filtering by company ticker.

    The CSV columns are expected to be::
        Question, Source Docs, Question Type, Source Chunk Type, Answer

    Returns a sampled DataFrame with at most *n_samples* rows.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required = {"Question", "Answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}. Found: {list(df.columns)}")

    df = df.dropna(subset=["Question", "Answer"])

    if company_filter:
        ticker = company_filter.upper()
        # Source Docs column contains entries like "*AAPL*" or "*AAPL*, *MSFT*"
        source_col = "Source Docs" if "Source Docs" in df.columns else None
        if source_col:
            df = df[df[source_col].str.contains(ticker, na=False, case=False)]
            if df.empty:
                raise ValueError(
                    f"No Q&A pairs found for company '{ticker}'. "
                    f"Unique values: {pd.read_csv(csv_path)['Source Docs'].unique()[:10].tolist()}"
                )

    if len(df) < n_samples:
        logger.warning(
            "Only %d Q&A pairs available (requested %d). Using all.", len(df), n_samples
        )

    return df.sample(n=min(n_samples, len(df)), random_state=random_seed).reset_index(drop=True)


def run_rag(question: str, config: Config) -> tuple[str, list[str]]:
    """
    Execute the full RAG pipeline for a single *question*.

    Returns:
        (generated_answer, retrieved_context_strings)
    """
    chunks = retrieve(
        query=question,
        config=config,
        top_k=config.retrieval.top_k,
    )
    generated_answer = generate(
        query=question,
        chunks=chunks,
        config=config.generation,
    )
    contexts = [c.text for c in chunks]
    return generated_answer, contexts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline using RAGAS + custom GEval metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=25,
        help="Number of Q&A pairs to evaluate (default: 25).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to save the per-sample results CSV (default: evaluation_results.csv).",
    )
    parser.add_argument(
        "--company",
        type=str,
        default=None,
        help="Filter Q&A pairs to a specific company ticker, e.g. AAPL.",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        default=False,
        help="Skip RAGAS LLM metrics (Correctness, Faithfulness, AnswerRelevancy). "
             "Useful for fast offline runs or cost control.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Q&A sampling (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()

    if not config.embedding.api_key:
        logger.error(
            "OPENAI_API_KEY is not set. "
            "Export it or add it to your .env file and retry."
        )
        sys.exit(1)

    # ── 1. Load golden Q&A pairs ───────────────────────────────────────────
    logger.info(
        "Loading up to %d Q&A pairs from %s (company=%s) ...",
        args.samples,
        QNA_CSV,
        args.company or "ALL",
    )
    qna_df = load_qna(
        QNA_CSV,
        n_samples=args.samples,
        company_filter=args.company,
        random_seed=args.seed,
    )
    logger.info("Loaded %d Q&A pairs.", len(qna_df))

    # ── 2. Run the RAG pipeline for each question ──────────────────────────
    eval_samples: list[EvalSample] = []
    failed = 0

    for idx, row in qna_df.iterrows():
        question = str(row["Question"]).strip()
        golden_answer = str(row["Answer"]).strip()
        current = len(eval_samples) + 1

        logger.info(
            "[%d/%d] RAG: %s ...",
            current,
            len(qna_df),
            question[:90],
        )

        try:
            generated_answer, contexts = run_rag(question, config)
        except Exception as exc:
            logger.warning("RAG pipeline failed for row %s: %s", idx, exc)
            failed += 1
            continue

        eval_samples.append(
            EvalSample(
                question=question,
                golden_answer=golden_answer,
                generated_answer=generated_answer,
                retrieved_contexts=contexts,
            )
        )

    if not eval_samples:
        logger.error("No successful RAG runs. Aborting evaluation.")
        sys.exit(1)

    if failed:
        logger.warning("%d question(s) failed during RAG and were skipped.", failed)

    # ── 3. Compute evaluation metrics ─────────────────────────────────────
    logger.info(
        "Computing metrics for %d samples (skip_ragas=%s) ...",
        len(eval_samples),
        args.skip_ragas,
    )
    results_df = evaluate_samples(
        samples=eval_samples,
        config=config,
        skip_ragas=args.skip_ragas,
    )

    # ── 4. Print summary ───────────────────────────────────────────────────
    print_summary(results_df)

    # ── 5. Save per-sample results ─────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info("Per-sample results saved to %s", output_path)

    # ── 6. Also save aggregated summary ───────────────────────────────────
    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    agg = (
        results_df[METRIC_COLUMNS]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .T.rename_axis("metric")
        .reset_index()
    )
    agg.to_csv(summary_path, index=False)
    logger.info("Aggregated summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
