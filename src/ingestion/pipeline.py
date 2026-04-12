"""
Orchestrates the full document ingestion pipeline:

    PDF files  →  parse  →  chunk  →  embed  →  Weaviate

Supports two chunking modes via the *use_smart* flag:

    use_smart=False  (default – Phase 1)
        • Token-based greedy chunker  (chunker.py)
        • Embeds raw chunk text
        • Stores in SecDocument collection

    use_smart=True   (Phase 2)
        • Parent-child chunker using RecursiveCharacterTextSplitter
          (smart_chunker.py)
        • Embeds context_window strings ("[COMPANY QX YEAR] Section: …\\n\\nchild")
          for stronger retrieval signal
        • Stores child chunks + parent_text in SecDocumentSmart collection

Each step is logged individually so you can observe where time is spent.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import Config
from src.ingestion import weaviate_store
from src.ingestion.chunker import chunk_documents
from src.ingestion.smart_chunker import chunk_documents_smart
from src.ingestion.embedder import create_embeddings
from src.ingestion.pdf_parser import parse_all_pdfs

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(
    docs_path: Optional[Path] = None,
    config: Optional[Config] = None,
    recreate_collection: bool = False,
    use_smart: bool = False,
    engine: str = "llamaindex",
) -> dict:
    """
    Run the end-to-end ingestion pipeline.

    Args:
        docs_path:           Override the docs directory from config.
        config:              Config object; uses defaults from env if None.
        recreate_collection: Drop & recreate the target collection before ingest.
        use_smart:           If True, use the parent-child smart chunker.
        engine:              "custom" (default) – custom chunker + Weaviate client.
                             "llamaindex"       – LlamaIndex SimpleDirectoryReader
                               + SentenceSplitter / HierarchicalNodeParser
                               + WeaviateVectorStore.

    Returns:
        Summary dict with counts for each stage.
    """
    if config is None:
        config = Config()
    if docs_path is not None:
        config.docs_path = Path(docs_path)

    if engine == "llamaindex":
        from src.ingestion.llamaindex_pipeline import run_llamaindex_pipeline
        return run_llamaindex_pipeline(
            docs_path=config.docs_path,
            config=config,
            use_smart=use_smart,
            recreate_collection=recreate_collection,
        )

    mode = "smart (parent-child)" if use_smart else "basic (token)"
    collection = (
        config.weaviate.smart_collection_name if use_smart
        else config.weaviate.collection_name
    )

    logger.info("=" * 60)
    logger.info("Starting SEC 10-Q ingestion pipeline  [mode=%s]", mode)
    logger.info("  docs_path  : %s", config.docs_path)
    if use_smart:
        logger.info("  parent     : %d tokens  overlap=%d",
                    config.smart_chunk.parent_max_tokens,
                    config.smart_chunk.parent_overlap_tokens)
        logger.info("  child      : %d tokens  overlap=%d",
                    config.smart_chunk.child_max_tokens,
                    config.smart_chunk.child_overlap_tokens)
    else:
        logger.info("  chunk size : %d–%d tokens  overlap=%d",
                    config.chunk.min_tokens, config.chunk.max_tokens,
                    config.chunk.overlap_tokens)
    logger.info("  embedding  : %s", config.embedding.model)
    logger.info("  collection : %s", collection)
    logger.info("=" * 60)

    # ── Step 1: Parse PDFs ────────────────────────────────────────────────────
    logger.info("[1/4] Parsing PDFs …")
    documents = parse_all_pdfs(config.docs_path)
    logger.info("      → %d documents parsed", len(documents))

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    logger.info("[2/4] Chunking documents  [mode=%s] …", mode)
    if use_smart:
        ingestion_ts = datetime.now(timezone.utc).isoformat()
        chunks = chunk_documents_smart(documents, config.smart_chunk, last_updated_date=ingestion_ts)
    else:
        chunks = chunk_documents(documents, config.chunk)
    logger.info("      → %d chunks created", len(chunks))

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    logger.info("[3/4] Creating embeddings via OpenAI …")
    if use_smart:
        # Embed context_window (child text + company/section prefix) so the
        # vector encodes filing identity for better retrieval precision.
        texts = [c.context_window for c in chunks]
    else:
        texts = [c.text for c in chunks]
    embeddings = create_embeddings(texts, config.embedding)
    logger.info("      → %d embedding vectors created", len(embeddings))

    # ── Step 4: Store in Weaviate ─────────────────────────────────────────────
    logger.info("[4/4] Storing in Weaviate (%s) …", collection)
    client = weaviate_store.get_client(config.weaviate)
    stored = 0
    total_in_db = 0
    try:
        if use_smart:
            if recreate_collection:
                weaviate_store.drop_collection_smart(client, config.weaviate)
            weaviate_store.ensure_schema_smart(client, config.weaviate)
            stored = weaviate_store.store_smart_chunks(client, chunks, embeddings, config.weaviate)
            total_in_db = weaviate_store.get_total_count_smart(client, config.weaviate)
        else:
            if recreate_collection:
                weaviate_store.drop_collection(client, config.weaviate)
            weaviate_store.ensure_schema(client, config.weaviate)
            stored = weaviate_store.store_chunks(client, chunks, embeddings, config.weaviate)
            total_in_db = weaviate_store.get_total_count(client, config.weaviate)
        logger.info("      → %d chunks stored  (total in DB: %d)", stored, total_in_db)
    finally:
        client.close()

    logger.info("=" * 60)
    logger.info("Pipeline complete  [mode=%s].", mode)
    logger.info("=" * 60)

    return {
        "documents_parsed": len(documents),
        "chunks_created":   len(chunks),
        "chunks_stored":    stored,
        "total_in_db":      total_in_db,
    }
