"""
Orchestrates the full document ingestion pipeline:

    PDF files  →  parse  →  chunk  →  embed  →  Weaviate

Each step is logged individually so you can observe where time is spent —
a key insight when evolving this from demo to production grade.
"""
import logging
from pathlib import Path
from typing import Optional

from src.config import Config
from src.ingestion import weaviate_store
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import create_embeddings
from src.ingestion.pdf_parser import parse_all_pdfs

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(
    docs_path: Optional[Path] = None,
    config: Optional[Config] = None,
    recreate_collection: bool = False,
) -> dict:
    """
    Run the end-to-end ingestion pipeline.

    Args:
        docs_path:           Override the docs directory from config.
        config:              Config object; uses defaults from env if None.
        recreate_collection: Drop & recreate Weaviate collection before ingest.

    Returns:
        Summary dict with counts for each stage.
    """
    if config is None:
        config = Config()
    if docs_path is not None:
        config.docs_path = Path(docs_path)

    logger.info("=" * 60)
    logger.info("Starting SEC 10-Q ingestion pipeline")
    logger.info("  docs_path  : %s", config.docs_path)
    logger.info("  chunk size : %d–%d tokens  overlap=%d",
                config.chunk.min_tokens, config.chunk.max_tokens,
                config.chunk.overlap_tokens)
    logger.info("  embedding  : %s", config.embedding.model)
    logger.info("  weaviate   : %s:%d", config.weaviate.http_host, config.weaviate.http_port)
    logger.info("=" * 60)

    # ── Step 1: Parse PDFs ────────────────────────────────────────────────────
    logger.info("[1/4] Parsing PDFs …")
    documents = parse_all_pdfs(config.docs_path)
    logger.info("      → %d documents parsed", len(documents))

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    logger.info("[2/4] Chunking documents …")
    chunks = chunk_documents(documents, config.chunk)
    logger.info("      → %d chunks created", len(chunks))

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    logger.info("[3/4] Creating embeddings via OpenAI …")
    texts = [c.text for c in chunks]
    embeddings = create_embeddings(texts, config.embedding)
    logger.info("      → %d embedding vectors created", len(embeddings))

    # ── Step 4: Store in Weaviate ─────────────────────────────────────────────
    logger.info("[4/4] Storing in Weaviate …")
    client = weaviate_store.get_client(config.weaviate)
    stored = 0
    total_in_db = 0
    try:
        if recreate_collection:
            weaviate_store.drop_collection(client, config.weaviate)
        weaviate_store.ensure_schema(client, config.weaviate)
        stored = weaviate_store.store_chunks(client, chunks, embeddings, config.weaviate)
        total_in_db = weaviate_store.get_total_count(client, config.weaviate)
        logger.info("      → %d chunks stored  (total in DB: %d)", stored, total_in_db)
    finally:
        client.close()

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)

    return {
        "documents_parsed": len(documents),
        "chunks_created": len(chunks),
        "chunks_stored": stored,
        "total_in_db": total_in_db,
    }
