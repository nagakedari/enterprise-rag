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

# Debug retrieval only — no generation, no metrics, fast
# Shows which chunks were retrieved and whether the gold source page was hit
python scripts/run_evaluation.py --debug --debug-limit 5
python scripts/run_evaluation.py --debug --debug-limit 10 --engine custom --use-smart --retrieval-mode hybrid

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
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# ── Project root on sys.path ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from src.config import Config
from src.evaluation.run_tag import build_run_tag, resolve_chunking_strategy
from src.retrieval.retriever import retrieve, retrieve_per_filing
from src.retrieval.llamaindex_retriever import retrieve_llamaindex
from src.retrieval.query_filters import extract_query_filters
from src.retrieval.reranker import rerank
from src.generation.generator import generate
from src.evaluation.evaluator import (
    ALL_METRICS,
    EvalSample,
    evaluate_samples,
    print_summary,
    build_hallucination_summary,
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

DEFAULT_QNA_CSV = Path(
    "/Users/manjusri/learning/generative_ai/KG-RAG-datasets/sec-10-q/data/v1/qna_data_with_page_numbers.csv"
)
DEFAULT_OUTPUT = Path("evaluation_results.csv")
DEBUG_OUTPUT_DIR = Path("/Users/manjusri/learning/generative_ai/enterprise-rag/debug/retrieval")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_qna(
    csv_path: Path,
    n_samples: int,
    company_filter: str | None = None,
    question_type_filter: str | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Load Q&A pairs from *csv_path*, optionally filtering by company ticker and/or
    question type (e.g. "Multi-Doc RAG", "Single-Doc Multi-Chunk RAG").

    The CSV columns are expected to be::
        Question, Source Docs, Question Type, Source Chunk Type, Answer

    Returns at most *n_samples* rows.  When the filtered dataset already has
    <= n_samples rows every row is returned in CSV order (no shuffling).
    Sampling only happens when the dataset is larger than n_samples.
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
        source_col = "Source Docs" if "Source Docs" in df.columns else None
        if source_col:
            df = df[df[source_col].str.contains(ticker, na=False, case=False)]
            if df.empty:
                raise ValueError(
                    f"No Q&A pairs found for company '{ticker}'. "
                    f"Unique values: {pd.read_csv(csv_path)['Source Docs'].unique()[:10].tolist()}"
                )

    if question_type_filter and "Question Type" in df.columns:
        df = df[df["Question Type"].str.strip() == question_type_filter]
        if df.empty:
            available = df["Question Type"].unique().tolist() if "Question Type" in df.columns else []
            raise ValueError(
                f"No Q&A pairs found for question type '{question_type_filter}'. "
                f"Available types: {available}"
            )
        logger.info("Filtered to question type '%s': %d rows.", question_type_filter, len(df))

    available = len(df)
    if available <= n_samples:
        # All rows fit — return in CSV order, no randomness involved.
        if available < n_samples:
            logger.warning(
                "Only %d Q&A pairs available (requested %d). Using all.", available, n_samples
            )
        return df.reset_index(drop=True)

    # Dataset is larger than requested — draw a reproducible random subset.
    logger.info("Sampling %d of %d Q&A pairs (seed=%d).", n_samples, available, random_seed)
    return df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Golden-answer cleaning (strip embedded source citations)
# ---------------------------------------------------------------------------

# Matches "(SOURCE: 2022 Q3 AAPL.pdf)" and "(SOURCE(S): file1.pdf, file2.pdf)"
# These citation artifacts appear in 193/195 golden answers and pollute token
# overlap metrics (exactness) because the generated answers never include them.
_CITATION_RE = re.compile(r"\s*\(SOURCE(?:\(S\))?:\s*[^)]+\)", re.IGNORECASE)


def _clean_golden(text: str) -> str:
    """Remove embedded source citations from a golden answer before metric computation."""
    return _CITATION_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Gold-page parsing helpers (used by debug mode)
# ---------------------------------------------------------------------------

def _parse_gold_pages(source_document_page: str) -> dict[str, set[int]]:
    """
    Parse the 'Source Document Page' column into {filename: {page_numbers}}.

    Column format (one PDF per line)::
        2022 Q3 AAPL.pdf, Pages 4, 10, 18 and 19
        2023 Q1 AAPL.pdf, Pages 4, 10, 19 and 20

    Returns::
        {"2022 Q3 AAPL.pdf": {4, 10, 18, 19}, "2023 Q1 AAPL.pdf": {4, 10, 19, 20}}
    """
    result: dict[str, set[int]] = {}
    if not source_document_page or pd.isna(source_document_page):
        return result
    for line in str(source_document_page).split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(.+\.pdf),\s*Pages?\s+(.+)$", line, re.IGNORECASE)
        if not m:
            continue
        filename = m.group(1).strip()
        pages_str = re.sub(r"\s+and\s+", ",", m.group(2))
        pages: set[int] = set()
        for part in pages_str.split(","):
            try:
                pages.add(int(part.strip()))
            except ValueError:
                pass
        if pages:
            result[filename] = pages
    return result


def _check_gold_match(chunks: list, gold_pages: dict[str, set[int]]) -> tuple[bool, str]:
    """
    Check whether any retrieved chunk falls on a gold source page.

    Returns (matched: bool, detail: str) where detail names the first hit or
    'no page info' when page numbers are unavailable (basic collection).
    """
    has_page_info = any(c.page_start > 0 for c in chunks)
    if not has_page_info:
        return False, "no page info (basic collection stores no page numbers)"

    for chunk in chunks:
        filename = chunk.source_file
        if filename in gold_pages:
            chunk_pages = set(range(chunk.page_start, chunk.page_end + 1))
            hit = chunk_pages & gold_pages[filename]
            if hit:
                return True, f"{filename} p{sorted(hit)[0]}"
    return False, "not found"


# ---------------------------------------------------------------------------
# Shared retrieval helper
# ---------------------------------------------------------------------------

def _retrieve_chunks(
    question: str,
    config: Config,
    engine: str = "custom",
    chunking_strategy: str = "basic",
    retrieval_mode: Optional[str] = None,
    retrieval_alpha: Optional[float] = None,
    filter_mode: str = "llm",
    rerank_mode: Optional[str] = None,
    rerank_top_k: Optional[int] = None,
    diversity_mode: Optional[str] = None,
    mmr_lambda: float = 0.5,
    per_filing: bool = False,
    chunks_per_filing: int = 3,
    max_per_entity: Optional[int] = None,
) -> list:
    """
    Run retrieval (with optional re-ranking) and return raw RetrievedChunk objects.
    Shared by run_rag() (which adds generation) and run_retrieval_only() (debug).

    per_filing: when True and the question has no quarter filter, runs a separate
    retrieval query per (year, quarter) filing for the company. Guarantees temporal
    coverage for questions like "How has Apple's net sales changed over time?"

    diversity_mode replaces the final "sort → top_k" step inside the reranker:
        "none"           – plain top-k sort (default)
        "mmr"            – Maximal Marginal Relevance (relevance + diversity)
        "metadata_slots" – guaranteed per-(company, quarter, year) slot coverage
        "source_cap"     – per-entity upper-bound cap (ceil(top_k / n_entities))
    """
    filters = extract_query_filters(
        question=question,
        mode=filter_mode,
        api_key=config.generation.api_key,
        model=config.generation.model,
    )
    logger.debug("Extracted filters: %s", filters)

    final_k = rerank_top_k if (rerank_mode and rerank_top_k is not None) else config.retrieval.top_k
    fetch_k = config.retrieval.top_k * 3 if rerank_mode else config.retrieval.top_k

    # Per-filing retrieval: triggered when question has a company but no quarter
    # (temporal question). Not supported for llamaindex engine.
    should_per_filing = (
        per_filing
        and filters.get("quarter") is None
        and filters.get("company") is not None
        and engine != "llamaindex"
    )

    if engine == "llamaindex":
        chunks = retrieve_llamaindex(
            query=question,
            config=config,
            top_k=fetch_k,
            use_smart=(chunking_strategy == "parent_child"),
            mode=retrieval_mode,
            alpha=retrieval_alpha,
            **filters,
        )
    elif should_per_filing:
        if chunking_strategy == "semantic":
            collection_name = config.weaviate.semantic_collection_name
        elif chunking_strategy == "parent_child":
            collection_name = config.weaviate.smart_collection_name
        else:
            collection_name = config.weaviate.collection_name
        chunks = retrieve_per_filing(
            query=question,
            config=config,
            company=filters["company"],
            chunks_per_filing=chunks_per_filing,
            collection_name=collection_name,
            mode=retrieval_mode,
            alpha=retrieval_alpha,
        )
        logger.info("Per-filing retrieval returned %d chunks", len(chunks))
    else:
        if chunking_strategy == "semantic":
            collection_name = config.weaviate.semantic_collection_name
        elif chunking_strategy == "parent_child":
            collection_name = config.weaviate.smart_collection_name
        else:
            collection_name = config.weaviate.collection_name
        chunks = retrieve(
            query=question,
            config=config,
            top_k=fetch_k,
            collection_name=collection_name,
            mode=retrieval_mode,
            alpha=retrieval_alpha,
            **filters,
        )

    if rerank_mode and len(chunks) > final_k:
        chunks = rerank(
            question=question,
            chunks=chunks,
            top_k=final_k,
            mode=rerank_mode,
            diversity_mode=diversity_mode or "none",
            mmr_lambda=mmr_lambda,
            max_per_entity=max_per_entity,
            api_key=config.generation.api_key,
            model=config.generation.model,
        )

    return chunks


def run_rag(
    question: str,
    config: Config,
    engine: str = "custom",
    use_smart: bool = False,
    chunking_strategy: Optional[str] = None,
    retrieval_mode: Optional[str] = None,
    retrieval_alpha: Optional[float] = None,
    filter_mode: str = "llm",
    rerank_mode: Optional[str] = None,
    rerank_top_k: Optional[int] = None,
    diversity_mode: Optional[str] = None,
    mmr_lambda: float = 0.5,
    per_filing: bool = False,
    chunks_per_filing: int = 3,
    max_per_entity: Optional[int] = None,
) -> tuple[str, list[str]]:
    """
    Execute the full RAG pipeline for a single *question*.

    Returns:
        (generated_answer, retrieved_context_strings)
    """
    if chunking_strategy is None:
        chunking_strategy = "parent_child" if use_smart else "basic"

    chunks = _retrieve_chunks(
        question, config, engine, chunking_strategy,
        retrieval_mode, retrieval_alpha, filter_mode, rerank_mode, rerank_top_k,
        diversity_mode, mmr_lambda, per_filing, chunks_per_filing, max_per_entity,
    )
    generated_answer = generate(
        query=question,
        chunks=chunks,
        config=config.generation,
    )
    return generated_answer, [c.text for c in chunks]


def run_retrieval_only(
    question: str,
    config: Config,
    engine: str = "custom",
    chunking_strategy: str = "basic",
    retrieval_mode: Optional[str] = None,
    retrieval_alpha: Optional[float] = None,
    filter_mode: str = "llm",
    rerank_mode: Optional[str] = None,
    rerank_top_k: Optional[int] = None,
    diversity_mode: Optional[str] = None,
    mmr_lambda: float = 0.5,
    per_filing: bool = False,
    chunks_per_filing: int = 3,
    max_per_entity: Optional[int] = None,
) -> list:
    """
    Run retrieval only — no generation, no LLM generation cost.
    Returns raw RetrievedChunk objects for inspection.
    Used by debug mode.
    """
    return _retrieve_chunks(
        question, config, engine, chunking_strategy,
        retrieval_mode, retrieval_alpha, filter_mode, rerank_mode, rerank_top_k,
        diversity_mode, mmr_lambda, per_filing, chunks_per_filing, max_per_entity,
    )


# ---------------------------------------------------------------------------
# Debug retrieval mode
# ---------------------------------------------------------------------------

def run_debug_mode(
    qna_df: pd.DataFrame,
    config: Config,
    args: argparse.Namespace,
    run_tag: str,
) -> None:
    """
    Retrieval-only debug run.  Generation is skipped entirely.

    For each question:
      - Runs retrieval (+ optional re-ranking)
      - Checks whether any retrieved chunk covers a gold source page
      - Prints a markdown table to the console
      - Saves a detailed CSV with per-chunk breakdown

    Output columns
    --------------
    question            Full question text
    gold_source_docs    Source Docs column value (e.g. *AAPL*)
    gold_pages_raw      Raw Source Document Page column value
    retrieved_chunks    Compact per-chunk summary (file, page, section, score)
    gold_page_present   ✅ / ❌ / ⚠ no page info
    match_detail        First matching file+page, or reason for miss
    """
    source_page_col = "Source Document Page"
    has_page_col = source_page_col in qna_df.columns

    rows = []
    for idx, row in qna_df.iterrows():
        question = str(row["Question"]).strip()
        source_docs = str(row.get("Source Docs", ""))
        gold_pages_raw = str(row.get(source_page_col, "")) if has_page_col else ""
        gold_pages = _parse_gold_pages(gold_pages_raw) if gold_pages_raw else {}
        current = len(rows) + 1

        logger.info(
            "[%d/%d] Debug retrieval: %s ...",
            current, len(qna_df), question[:80],
        )

        try:
            chunks = run_retrieval_only(
                question,
                config,
                engine=args.engine,
                chunking_strategy=args.chunking_strategy or ("parent_child" if args.use_smart else "basic"),
                retrieval_mode=args.retrieval_mode,
                retrieval_alpha=args.alpha,
                filter_mode=args.filter_mode,
                rerank_mode=args.rerank_mode,
                rerank_top_k=args.rerank_top_k,
                diversity_mode=args.diversity_mode,
                mmr_lambda=args.mmr_lambda,
                per_filing=args.per_filing,
                chunks_per_filing=args.chunks_per_filing,
                max_per_entity=args.max_per_entity,
            )
        except Exception as exc:
            logger.warning("Retrieval failed for row %s: %s", idx, exc, exc_info=True)
            rows.append({
                "question": question,
                "gold_source_docs": source_docs,
                "gold_pages_raw": gold_pages_raw,
                "retrieved_chunks": "ERROR",
                "gold_page_present": "❌ error",
                "match_detail": str(exc),
            })
            continue

        # Format each retrieved chunk as a compact label
        chunk_labels = []
        for i, c in enumerate(chunks):
            page = f"p{c.page_start}" if c.page_start == c.page_end else f"p{c.page_start}-{c.page_end}"
            section = f" §{c.section_title[:25]}" if getattr(c, "section_title", "") else ""
            chunk_labels.append(f"[{i+1}] {c.source_file} {page}{section} (score={c.score:.3f})")

        # Gold page match check
        if gold_pages:
            matched, detail = _check_gold_match(chunks, gold_pages)
            if "no page info" in detail:
                present_symbol = "⚠"
            else:
                present_symbol = "✅" if matched else "❌"
        else:
            present_symbol, detail = "⚠", "no gold page data in CSV"

        # Diversity: unique source files and unique section titles in retrieved set
        unique_files    = sorted({c.source_file for c in chunks})
        unique_sections = sorted({
            (getattr(c, "section_title", "") or "").strip()[:40]
            for c in chunks
            if (getattr(c, "section_title", "") or "").strip()
        })

        # How many gold source files do we need vs how many we actually retrieved?
        gold_file_count      = len(gold_pages) if gold_pages else 0
        retrieved_file_count = len(unique_files)

        rows.append({
            "question":            question,
            "gold_source_docs":    source_docs,
            "gold_pages_raw":      gold_pages_raw,
            "retrieved_chunks":    " | ".join(chunk_labels),
            "gold_page_present":   present_symbol,
            "match_detail":        detail,
            "unique_files_retrieved": retrieved_file_count,
            "gold_files_needed":   gold_file_count,
            "unique_sections":     "; ".join(unique_sections) if unique_sections else "(none)",
            # Full chunk texts for content inspection
            "chunk_texts": [c.text for c in chunks],
        })

    debug_df = pd.DataFrame(rows)

    # ── Print markdown table to console ──────────────────────────────────────
    print()
    print("=" * 90)
    print(f"  RETRIEVAL DEBUG  ({len(rows)} questions)  [{run_tag}]")
    print("=" * 90)
    print()
    print(f"| {'#':>3} | {'Question':<45} | {'Files: Got/Need':^15} | {'Sections':^25} | {'Gold Page?':^10} |")
    print(f"|{'-'*5}|{'-'*47}|{'-'*17}|{'-'*27}|{'-'*12}|")
    for i, r in debug_df.iterrows():
        q_short   = r["question"][:44].replace("|", "\\|")
        file_ratio = f"{r['unique_files_retrieved']}/{r['gold_files_needed'] or '?'}"
        secs_short = (r["unique_sections"] or "")[:24]
        print(f"| {i+1:>3} | {q_short:<45} | {file_ratio:^15} | {secs_short:<25} | {r['gold_page_present']:^10} |")

    print()
    total         = len(rows)
    matched_count = sum(1 for r in rows if r["gold_page_present"] == "✅")
    missed_count  = sum(1 for r in rows if r["gold_page_present"] == "❌")
    warn_count    = sum(1 for r in rows if r["gold_page_present"] == "⚠")
    print(f"  ✅ Gold page hit : {matched_count}/{total}")
    print(f"  ❌ Gold page miss: {missed_count}/{total}")
    if warn_count:
        print(f"  ⚠  No page data  : {warn_count}/{total}")
    print()

    # ── Per-question chunk detail ─────────────────────────────────────────────
    for i, r in debug_df.iterrows():
        print(f"  {'─'*80}")
        print(f"  Q{i+1}: {r['question'][:100]}")
        print(f"  Gold files needed : {r['gold_pages_raw'][:120] if r['gold_pages_raw'] else 'n/a'}")
        chunk_texts = r.get("chunk_texts", [])
        for j, (lbl, txt) in enumerate(zip(r["retrieved_chunks"].split(" | "), chunk_texts)):
            preview = txt.replace("\n", " ").strip()[:180] if txt else ""
            print(f"    [{j+1}] {lbl}")
            print(f"         ↳ {preview!r}")
        print()

    # ── Save detailed CSV (drop the list column — not CSV-friendly) ──────────
    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    debug_path = DEBUG_OUTPUT_DIR / f"{run_tag}_retrieval_debug.csv"
    debug_df.drop(columns=["chunk_texts"], errors="ignore").to_csv(debug_path, index=False)
    logger.info("Retrieval debug table saved to %s", debug_path)


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
        "--input",
        type=str,
        default=None,
        help=(
            "Path to the golden Q&A CSV file (relative or absolute). "
            f"Defaults to {DEFAULT_QNA_CSV}. "
            "Required columns: Question, Answer. "
            "Optional columns: Source Docs, Source Document Page."
        ),
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
        "--question-type",
        type=str,
        default=None,
        dest="question_type",
        help=(
            "Filter Q&A pairs to a specific question type. "
            "Valid values: 'Multi-Doc RAG', 'Single-Doc Single-Chunk RAG', "
            "'Single-Doc Multi-Chunk RAG'. "
            "Useful with --debug to target a specific retrieval scenario."
        ),
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
    parser.add_argument(
        "--engine",
        type=str,
        default="custom",
        choices=["custom", "llamaindex"],
        help="Retrieval engine to evaluate (default: custom).",
    )
    parser.add_argument(
        "--use-smart",
        action="store_true",
        default=False,
        help="Legacy flag: use parent-child collection. Prefer --chunking-strategy parent_child.",
    )
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        default=None,
        choices=["basic", "parent_child", "semantic"],
        help=(
            "Which ingested collection to query: "
            "'basic' → SecDocument, "
            "'parent_child' → SecDocumentSmart, "
            "'semantic' → DocumentChunk (BGE-M3). "
            "Overrides --use-smart when provided."
        ),
    )
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default=None,
        choices=["semantic", "hybrid"],
        help="Override retrieval mode for this run (default: uses RETRIEVAL_MODE env var).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="BM25/vector balance for hybrid mode: 0.0=BM25 only, 1.0=vector only (default: config).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        dest="top_k",
        help=(
            "Number of chunks to retrieve per question (default: RETRIEVAL_TOP_K env var, "
            "usually 5). Raise to 10-15 for Multi-Doc RAG questions that span multiple quarters."
        ),
    )
    parser.add_argument(
        "--filter-mode",
        type=str,
        default="llm",
        choices=["llm", "regex"],
        help=(
            "How to extract company/year/quarter filters from each question. "
            "'llm' (default): one gpt-4o-mini call per question, handles arbitrary phrasing. "
            "'regex': zero LLM cost, covers explicit patterns like 'Apple Q2 2023'."
        ),
    )
    parser.add_argument(
        "--rerank-mode",
        type=str,
        default=None,
        choices=["llm", "cross_encoder"],
        help=(
            "Re-rank the over-fetched candidates before generation. "
            "'llm': one gpt-4o-mini call per question ranks all candidates (default when enabled). "
            "'cross_encoder': local sentence-transformers model, no API cost "
            "(requires: pip install sentence-transformers). "
            "Omit to skip re-ranking entirely."
        ),
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=None,
        dest="rerank_top_k",
        help=(
            "Number of chunks to keep AFTER re-ranking (passed to the LLM for generation). "
            "Defaults to --top-k when omitted. Set lower than --top-k to fetch a wide "
            "candidate pool for good recall, then aggressively prune noise for better "
            "precision and lower hallucination. "
            "Example: --top-k 10 --rerank-top-k 3"
        ),
    )
    parser.add_argument(
        "--diversity-mode",
        type=str,
        default="none",
        choices=["none", "mmr", "metadata_slots", "source_cap"],
        dest="diversity_mode",
        help=(
            "Final chunk selection strategy after scoring — replaces the plain "
            "top-k sort inside the reranker. Requires --rerank-mode to be set. "
            "'none' (default): sort by score, take top --rerank-top-k. "
            "'mmr': Maximal Marginal Relevance — iteratively picks chunks that are "
            "relevant to the query AND dissimilar to already-chosen chunks. Reduces "
            "redundant table rows; improves coverage for multi-chunk questions. "
            "Requires scikit-learn (pip install scikit-learn). "
            "'metadata_slots': guarantees proportional slot coverage per "
            "(company, quarter, year) entity in the candidate pool. Prevents the "
            "reranker from filling all slots with one quarter on Multi-Doc questions."
        ),
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.5,
        dest="mmr_lambda",
        help=(
            "MMR relevance/diversity trade-off (only used with --diversity-mode mmr). "
            "1.0 = pure relevance (same as top-k sort). "
            "0.0 = pure diversity (ignores relevance scores). "
            "Default: 0.5 (equal balance). "
            "Try 0.7 to bias toward relevance while still penalising redundant chunks."
        ),
    )
    parser.add_argument(
        "--per-filing",
        action="store_true",
        default=False,
        dest="per_filing",
        help=(
            "Run one retrieval query per (year, quarter) filing for the company "
            "when the question has no quarter filter. Guarantees temporal coverage "
            "for questions like 'How has Apple's net sales changed over time?' "
            "Ignored when a quarter is extracted from the question or when "
            "--engine=llamaindex."
        ),
    )
    parser.add_argument(
        "--chunks-per-filing",
        type=int,
        default=3,
        dest="chunks_per_filing",
        help=(
            "Number of chunks to fetch per (year, quarter) filing when --per-filing "
            "is active (default: 3). The total candidate pool is "
            "n_filings × chunks_per_filing before reranking."
        ),
    )
    parser.add_argument(
        "--max-per-entity",
        type=int,
        default=None,
        dest="max_per_entity",
        help=(
            "Override for the source_cap per-(company, quarter, year) slot cap "
            "(only used with --diversity-mode source_cap). "
            "Default: ceil(top_k / n_entities), minimum 2."
        ),
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help=(
            "Comma-separated list of metrics to compute. "
            "If omitted, all metrics are computed. "
            "Valid names: "
            + ", ".join(sorted(ALL_METRICS))
            + ". "
            "Example: --metrics exactness,faithfulness,context_precision"
        ),
    )
    # ── Debug mode ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help=(
            "Retrieval debug mode: run retrieval only, skip generation and metrics. "
            "Outputs a table showing retrieved chunks and whether the gold source page "
            "was present. No LLM generation calls are made (saves tokens)."
        ),
    )
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=5,
        help=(
            "Number of questions to evaluate in --debug mode (default: 5). "
            "Ignored when --debug is not set; use --samples for full evaluation runs."
        ),
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

    if args.top_k is not None:
        config.retrieval.top_k = args.top_k
        logger.info("top_k overridden to %d via --top-k", args.top_k)

    # Resolve effective strategy for run tag and pipeline calls
    chunking_strategy = resolve_chunking_strategy(args.chunking_strategy, args.use_smart)

    # Build a run tag for output filenames so different configs don't overwrite each other
    # e.g. "custom_semantic_hybrid_a0.5_regexfilters_cross_encoderrerank"
    mode_tag = args.retrieval_mode or config.retrieval.mode
    run_tag = build_run_tag(
        engine=args.engine,
        chunking_strategy=chunking_strategy,
        filter_mode=args.filter_mode,
        config_retrieval_mode=config.retrieval.mode,
        retrieval_mode=args.retrieval_mode,
        alpha=args.alpha,
        top_k=args.top_k,
        rerank_mode=args.rerank_mode,
        rerank_top_k=args.rerank_top_k,
        diversity_mode=args.diversity_mode,
        mmr_lambda=args.mmr_lambda,
        per_filing=args.per_filing,
        chunks_per_filing=args.chunks_per_filing,
        max_per_entity=args.max_per_entity,
        question_type=args.question_type,
        seed=args.seed,
    )

    logger.info(
        "Evaluation config: engine=%s  chunking_strategy=%s  mode=%s  alpha=%s",
        args.engine, chunking_strategy, mode_tag, args.alpha,
    )

    # ── 1. Load golden Q&A pairs ───────────────────────────────────────────
    qna_csv = Path(args.input).resolve() if args.input else DEFAULT_QNA_CSV
    if not qna_csv.exists():
        logger.error("Input file not found: %s", qna_csv)
        sys.exit(1)

    n_samples = args.debug_limit if args.debug else args.samples
    logger.info(
        "Input file : %s", qna_csv
    )
    logger.info(
        "Loading up to %d Q&A pairs (company=%s, question_type=%s) ...",
        n_samples,
        args.company or "ALL",
        args.question_type or "ALL",
    )
    qna_df = load_qna(
        qna_csv,
        n_samples=n_samples,
        company_filter=args.company,
        question_type_filter=args.question_type,
        random_seed=args.seed,
    )
    logger.info("Loaded %d Q&A pairs.", len(qna_df))

    # ── Debug mode: retrieval only, no generation ──────────────────────────
    if args.debug:
        logger.info("Running in DEBUG mode — retrieval only, generation skipped.")
        run_debug_mode(qna_df, config, args, run_tag)
        return

    # ── 2. Run the RAG pipeline for each question ──────────────────────────
    eval_samples: list[EvalSample] = []
    failed = 0
    total = len(qna_df)

    for row_num, (idx, row) in enumerate(qna_df.iterrows(), start=1):
        question = str(row["Question"]).strip()
        golden_answer = _clean_golden(str(row["Answer"]).strip())
        question_type = str(row.get("Question Type", ""))
        source_chunk_type = str(row.get("Source Chunk Type", ""))

        logger.info(
            "[%d/%d] RAG [%s]: %s ...",
            row_num,
            total,
            run_tag,
            question[:80],
        )

        try:
            generated_answer, contexts = run_rag(
                question,
                config,
                engine=args.engine,
                chunking_strategy=chunking_strategy,
                retrieval_mode=args.retrieval_mode,
                retrieval_alpha=args.alpha,
                filter_mode=args.filter_mode,
                rerank_mode=args.rerank_mode,
                rerank_top_k=args.rerank_top_k,
                diversity_mode=args.diversity_mode,
                mmr_lambda=args.mmr_lambda,
                per_filing=args.per_filing,
                chunks_per_filing=args.chunks_per_filing,
                max_per_entity=args.max_per_entity,
            )
        except Exception as exc:
            logger.warning("RAG pipeline failed for row %s: %s", idx, exc, exc_info=True)
            failed += 1
            continue

        eval_samples.append(
            EvalSample(
                question=question,
                golden_answer=golden_answer,
                generated_answer=generated_answer,
                retrieved_contexts=contexts,
                question_type=question_type,
                source_chunk_type=source_chunk_type,
            )
        )

    if not eval_samples:
        logger.error("No successful RAG runs. Aborting evaluation.")
        sys.exit(1)

    if failed:
        logger.warning("%d question(s) failed during RAG and were skipped.", failed)

    # ── 3. Compute evaluation metrics ─────────────────────────────────────
    # Parse and validate the --metrics filter (None = compute everything)
    requested_metrics = None
    if args.metrics:
        requested_metrics = {m.strip() for m in args.metrics.split(",") if m.strip()}
        unknown = requested_metrics - ALL_METRICS
        if unknown:
            logger.warning(
                "Unknown metric name(s) will be ignored: %s  "
                "Valid names: %s",
                sorted(unknown), sorted(ALL_METRICS),
            )
            requested_metrics -= unknown
        if not requested_metrics:
            logger.error("No valid metric names provided. Aborting.")
            sys.exit(1)
        logger.info("Selective evaluation — computing only: %s", sorted(requested_metrics))

    logger.info(
        "Computing metrics for %d samples (skip_ragas=%s) ...",
        len(eval_samples),
        args.skip_ragas,
    )
    results_df = evaluate_samples(
        samples=eval_samples,
        config=config,
        skip_ragas=args.skip_ragas,
        metrics=requested_metrics,
    )

    # ── 4. Print summary ───────────────────────────────────────────────────
    print_summary(results_df)

    # ── 5. Save per-sample results ─────────────────────────────────────────
    # Embed run_tag into the filename so different configs don't overwrite each other
    # e.g. evaluation_results_custom_smart_hybrid_a0.5.csv
    base_path = Path(args.output)
    output_path = base_path.with_name(f"{base_path.stem}_{run_tag}{base_path.suffix}")
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
    hal_summary = build_hallucination_summary(results_df)
    hal_rows = hal_summary.rename(columns={"avg_score": "mean"})[
        ["metric", "mean", "avg_pct", "flagged_count", "total", "flagged_pct", "threshold", "note"]
    ]
    agg = pd.concat([agg, hal_rows], ignore_index=True)
    # Stamp the input file and run config into the summary so every result CSV
    # is self-documenting — prevents the "which file did this run use?" ambiguity.
    meta = pd.DataFrame([{
        "metric": "_meta_input_file",   "mean": str(qna_csv),
    }, {
        "metric": "_meta_run_tag",      "mean": run_tag,
    }, {
        "metric": "_meta_n_questions",  "mean": len(results_df),
    }])
    agg = pd.concat([agg, meta], ignore_index=True)
    agg.to_csv(summary_path, index=False)
    logger.info("Aggregated summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
