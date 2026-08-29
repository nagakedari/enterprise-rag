#!/usr/bin/env python3
"""
Run the SEC 10-Q ingestion pipeline directly — no Airflow required.

Usage:
    # Use defaults from .env  (custom engine, basic chunking)
    python scripts/run_ingestion.py

    # Parent-child (smart) chunking
    python scripts/run_ingestion.py --chunking-strategy parent_child

    # Semantic chunking with BGE-M3 (Phase 3)
    python scripts/run_ingestion.py --chunking-strategy semantic

    # LlamaIndex engine with parent-child chunking
    python scripts/run_ingestion.py --engine llamaindex --chunking-strategy parent_child

    # Clean re-ingest with semantic chunking
    python scripts/run_ingestion.py --chunking-strategy semantic --recreate

    # Semantic chunking with checkpointing (resume if interrupted)
    python scripts/run_ingestion.py --chunking-strategy semantic --checkpoint-dir ./checkpoints

    # Override docs path
    python scripts/run_ingestion.py --docs-path /path/to/pdfs

Legacy flag (still supported):
    python scripts/run_ingestion.py --use-smart   # same as --chunking-strategy parent_child
"""
import argparse
import logging
import sys
from pathlib import Path

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest SEC 10-Q PDFs into Weaviate vector DB"
    )
    parser.add_argument(
        "--docs-path",
        type=Path,
        default=None,
        help="Directory containing PDF files (overrides DOCS_PATH env var)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Weaviate collection before ingestion",
    )
    parser.add_argument(
        "--engine",
        choices=["custom", "llamaindex"],
        default="custom",
        help="Ingestion engine: 'custom' (default) or 'llamaindex'",
    )
    parser.add_argument(
        "--use-smart",
        action="store_true",
        help="Legacy flag: use parent-child chunker. Prefer --chunking-strategy parent_child.",
    )
    parser.add_argument(
        "--chunking-strategy",
        choices=["basic", "parent_child", "semantic"],
        default=None,
        help=(
            "Chunking strategy to use: "
            "'basic' – token-based greedy chunker → SecDocument. "
            "'parent_child' – parent-child hierarchical chunker → SecDocumentSmart. "
            "'semantic' – SemanticChunker with BGE-M3 embeddings → DocumentChunk. "
            "When omitted, falls back to --use-smart for compatibility."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory for checkpoint files.  When set, each expensive step "
            "(parse, chunk, embed) saves its output so a subsequent run can "
            "resume from where it was interrupted instead of starting over. "
            "Example: --checkpoint-dir ./checkpoints"
        ),
    )
    args = parser.parse_args()

    from src.ingestion.pipeline import run_ingestion_pipeline

    result = run_ingestion_pipeline(
        docs_path=args.docs_path,
        engine=args.engine,
        use_smart=args.use_smart,
        chunking_strategy=args.chunking_strategy,
        recreate_collection=args.recreate,
        checkpoint_dir=args.checkpoint_dir,
    )

    print("\n" + "=" * 50)
    print("  Ingestion complete")
    print("=" * 50)
    print(f"  Documents parsed  : {result['documents_parsed']}")
    print(f"  Chunks created    : {result['chunks_created']}")
    print(f"  Chunks stored     : {result['chunks_stored']}")
    print(f"  Total in DB       : {result['total_in_db']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
