#!/usr/bin/env python3
"""
Run the SEC 10-Q ingestion pipeline directly — no Airflow required.

Usage:
    # Use defaults from .env  (custom engine, basic chunking)
    python scripts/run_ingestion.py

    # LlamaIndex engine with smart (parent-child) chunking
    python scripts/run_ingestion.py --engine llamaindex --use-smart

    # Custom engine with smart chunking, clean re-ingest
    python scripts/run_ingestion.py --use-smart --recreate

    # Override docs path
    python scripts/run_ingestion.py --docs-path /path/to/pdfs
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
        help="Use parent-child smart chunker (SecDocumentSmart collection)",
    )
    args = parser.parse_args()

    from src.ingestion.pipeline import run_ingestion_pipeline

    result = run_ingestion_pipeline(
        docs_path=args.docs_path,
        engine=args.engine,
        use_smart=args.use_smart,
        recreate_collection=args.recreate,
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
