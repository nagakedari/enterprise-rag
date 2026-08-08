"""
Orchestrates the full document ingestion pipeline:

    PDF files  →  parse  →  chunk  →  embed  →  Weaviate

Supports three chunking strategies via the *chunking_strategy* parameter
(or the legacy *use_smart* boolean for backward compatibility):

    chunking_strategy="basic"  (Phase 1)
        • Token-based greedy chunker  (chunker.py)
        • Embeds raw chunk text with OpenAI
        • Stores in SecDocument collection

    chunking_strategy="parent_child"  (Phase 2, formerly use_smart=True)
        • Parent-child chunker using RecursiveCharacterTextSplitter
          (smart_chunker.py)
        • Embeds context_window strings ("[COMPANY QX YEAR] Section: …\\n\\nchild")
          for stronger retrieval signal
        • Stores child chunks + parent_text in SecDocumentSmart collection

    chunking_strategy="semantic"  (Phase 3)
        • SemanticChunker with BGE-M3 embeddings (semantic_chunker.py)
        • Splits at topically coherent boundaries, not token counts
        • Same BGE-M3 model used for both chunking and retrieval embeddings
        • Stores dual-field chunks (text_for_search + raw_content) in DocumentChunk

Each step is logged individually so you can observe where time is spent.
"""
import logging
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.config import Config
from src.ingestion import weaviate_store
from src.ingestion.chunker import chunk_document
from src.ingestion.smart_chunker import chunk_document_smart
from src.ingestion.semantic_chunker import chunk_document_semantic, create_bge_embeddings
from src.ingestion.embedder import create_embeddings
from src.ingestion.pdf_parser import parse_all_pdfs

logger = logging.getLogger(__name__)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _save_checkpoint(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Checkpoint saved → %s", path)


def _load_checkpoint(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_embeddings_ckpt(path: Path, embeddings: List[List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), np.array(embeddings, dtype="float32"))
    logger.info("Checkpoint saved → %s", path)


def _load_embeddings_ckpt(path: Path) -> List[List[float]]:
    return np.load(str(path)).tolist()


def run_ingestion_pipeline(
    docs_path: Optional[Path] = None,
    config: Optional[Config] = None,
    recreate_collection: bool = False,
    use_smart: bool = False,
    engine: str = "custom",
    chunking_strategy: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
) -> dict:
    """
    Run the end-to-end ingestion pipeline.

    Args:
        docs_path:           Override the docs directory from config.
        config:              Config object; uses defaults from env if None.
        recreate_collection: Drop & recreate the target collection before ingest.
        use_smart:           Legacy flag. True maps to chunking_strategy="parent_child".
                             Ignored when chunking_strategy is provided explicitly.
        engine:              "custom" (default) – custom chunker + Weaviate client.
                             "llamaindex"       – LlamaIndex SimpleDirectoryReader
                               + SentenceSplitter / HierarchicalNodeParser
                               + WeaviateVectorStore.
        chunking_strategy:   "basic"        – token-based greedy chunker (Phase 1).
                             "parent_child" – parent-child chunker (Phase 2).
                             "semantic"     – SemanticChunker with BGE-M3 (Phase 3).
                             None           – falls back to use_smart for compatibility.
        checkpoint_dir:      Directory for per-PDF checkpoint files.  When set,
                             chunks and embeddings are saved after each PDF so a
                             subsequent run skips already-processed PDFs.
                             Layout: {checkpoint_dir}/{strategy}/chunks/{stem}.pkl
                                     {checkpoint_dir}/{strategy}/embeddings/{stem}.npy

    Returns:
        Summary dict with counts for each stage.
    """
    if config is None:
        config = Config()
    if docs_path is not None:
        config.docs_path = Path(docs_path)

    # Resolve effective strategy: explicit chunking_strategy wins over legacy use_smart
    if chunking_strategy is None:
        chunking_strategy = "parent_child" if use_smart else "basic"

    if engine == "llamaindex":
        from src.ingestion.llamaindex_pipeline import run_llamaindex_pipeline
        return run_llamaindex_pipeline(
            docs_path=config.docs_path,
            config=config,
            use_smart=(chunking_strategy == "parent_child"),
            recreate_collection=recreate_collection,
        )

    strategy_labels = {
        "basic":        "basic (token)",
        "parent_child": "smart (parent-child)",
        "semantic":     "semantic (BGE-M3)",
    }
    mode = strategy_labels.get(chunking_strategy, chunking_strategy)

    collection_map = {
        "basic":        config.weaviate.collection_name,
        "parent_child": config.weaviate.smart_collection_name,
        "semantic":     config.weaviate.semantic_collection_name,
    }
    collection = collection_map.get(chunking_strategy, config.weaviate.collection_name)

    logger.info("=" * 60)
    logger.info("Starting SEC 10-Q ingestion pipeline  [strategy=%s]", chunking_strategy)
    logger.info("  docs_path  : %s", config.docs_path)
    if chunking_strategy == "parent_child":
        logger.info("  parent     : %d tokens  overlap=%d",
                    config.smart_chunk.parent_max_tokens,
                    config.smart_chunk.parent_overlap_tokens)
        logger.info("  child      : %d tokens  overlap=%d",
                    config.smart_chunk.child_max_tokens,
                    config.smart_chunk.child_overlap_tokens)
    elif chunking_strategy == "semantic":
        logger.info("  model      : %s", config.semantic_chunk.embedding_model)
        logger.info("  breakpoint : type=%s  amount=%s",
                    config.semantic_chunk.breakpoint_threshold_type,
                    config.semantic_chunk.breakpoint_threshold_amount)
        logger.info("  ceiling    : %d tokens  min_chars=%d  overlap=%d chars",
                    config.semantic_chunk.token_ceiling,
                    config.semantic_chunk.min_char_size,
                    config.semantic_chunk.overlap_chars)
    else:
        logger.info("  chunk size : %d–%d tokens  overlap=%d",
                    config.chunk.min_tokens, config.chunk.max_tokens,
                    config.chunk.overlap_tokens)
    logger.info("  collection : %s", collection)
    logger.info("=" * 60)

    # ── Per-PDF checkpoint directories ────────────────────────────────────────
    ckpt: Optional[Path] = Path(checkpoint_dir) if checkpoint_dir else None
    chunks_ckpt_dir  = ckpt / chunking_strategy / "chunks"     if ckpt else None
    embeds_ckpt_dir  = ckpt / chunking_strategy / "embeddings" if ckpt else None
    if chunks_ckpt_dir:
        chunks_ckpt_dir.mkdir(parents=True, exist_ok=True)
    if embeds_ckpt_dir:
        embeds_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Parse PDFs ────────────────────────────────────────────────────
    logger.info("[1/4] Parsing PDFs …")
    documents = parse_all_pdfs(config.docs_path)
    logger.info("      → %d documents parsed", len(documents))

    # ── Step 2: Chunk per PDF ─────────────────────────────────────────────────
    logger.info("[2/4] Chunking documents  [mode=%s] …", mode)
    ingestion_ts = datetime.now(timezone.utc).isoformat()
    chunks: list = []
    n_chunk_cached = 0

    for doc in documents:
        pdf_stem = Path(doc.source_file).stem
        pdf_chunks_ckpt = chunks_ckpt_dir / f"{pdf_stem}.pkl" if chunks_ckpt_dir else None

        if pdf_chunks_ckpt and pdf_chunks_ckpt.exists():
            doc_chunks = _load_checkpoint(pdf_chunks_ckpt)
            n_chunk_cached += 1
            logger.debug("  [chunk cache] %s → %d chunks", pdf_stem, len(doc_chunks))
        else:
            if chunking_strategy == "parent_child":
                doc_chunks = chunk_document_smart(doc, config.smart_chunk,
                                                  last_updated_date=ingestion_ts)
            elif chunking_strategy == "semantic":
                doc_chunks = chunk_document_semantic(doc, config.semantic_chunk,
                                                     last_updated_date=ingestion_ts)
            else:
                doc_chunks = chunk_document(doc, config.chunk)
            if pdf_chunks_ckpt:
                _save_checkpoint(pdf_chunks_ckpt, doc_chunks)
            logger.info("  [chunked]     %s → %d chunks", pdf_stem, len(doc_chunks))

        chunks.extend(doc_chunks)

    logger.info(
        "      → %d chunks total  (%d/%d PDFs from cache)",
        len(chunks), n_chunk_cached, len(documents),
    )

    # ── Step 3: Embed per PDF ─────────────────────────────────────────────────
    if chunking_strategy == "semantic":
        backend = "ONNX" if config.semantic_chunk.use_onnx else "PyTorch"
        logger.info("[3/4] Creating embeddings via BGE-M3 (%s) …", backend)
    else:
        logger.info("[3/4] Creating embeddings via OpenAI …")

    # Group chunk indices by source PDF to allow per-PDF cache lookup
    indices_by_pdf: dict = defaultdict(list)
    for i, chunk in enumerate(chunks):
        indices_by_pdf[Path(chunk.source_file).stem].append(i)

    embeddings: list = [None] * len(chunks)
    n_emb_cached = 0

    for pdf_stem, idx_list in indices_by_pdf.items():
        pdf_emb_ckpt = embeds_ckpt_dir / f"{pdf_stem}.npy" if embeds_ckpt_dir else None

        if pdf_emb_ckpt and pdf_emb_ckpt.exists():
            doc_embeddings = _load_embeddings_ckpt(pdf_emb_ckpt)
            n_emb_cached += 1
            logger.debug("  [emb cache]   %s → %d vectors", pdf_stem, len(doc_embeddings))
        else:
            pdf_chunks = [chunks[i] for i in idx_list]
            if chunking_strategy == "semantic":
                texts = [c.text_for_search for c in pdf_chunks]
                doc_embeddings = create_bge_embeddings(
                    texts,
                    config.semantic_chunk.embedding_model,
                    config.semantic_chunk.use_onnx,
                )
            elif chunking_strategy == "parent_child":
                texts = [c.context_window for c in pdf_chunks]
                doc_embeddings = create_embeddings(texts, config.embedding)
            else:
                texts = [c.text for c in pdf_chunks]
                doc_embeddings = create_embeddings(texts, config.embedding)
            if pdf_emb_ckpt:
                _save_embeddings_ckpt(pdf_emb_ckpt, doc_embeddings)
            logger.info("  [embedded]    %s → %d vectors", pdf_stem, len(doc_embeddings))

        for i, emb in zip(idx_list, doc_embeddings):
            embeddings[i] = emb

    logger.info(
        "      → %d vectors total  (%d/%d PDFs from cache)",
        len(embeddings), n_emb_cached, len(indices_by_pdf),
    )

    # ── Step 4: Store in Weaviate ─────────────────────────────────────────────
    logger.info("[4/4] Storing in Weaviate (%s) …", collection)
    client = weaviate_store.get_client(config.weaviate)
    stored = 0
    total_in_db = 0
    try:
        if chunking_strategy == "parent_child":
            if recreate_collection:
                weaviate_store.drop_collection_smart(client, config.weaviate)
            weaviate_store.ensure_schema_smart(client, config.weaviate)
            stored = weaviate_store.store_smart_chunks(client, chunks, embeddings, config.weaviate)
            total_in_db = weaviate_store.get_total_count_smart(client, config.weaviate)
        elif chunking_strategy == "semantic":
            if recreate_collection:
                weaviate_store.drop_collection_semantic(client, config.weaviate)
            weaviate_store.ensure_schema_semantic(client, config.weaviate)
            stored = weaviate_store.store_semantic_chunks(client, chunks, embeddings, config.weaviate)
            total_in_db = weaviate_store.get_total_count_semantic(client, config.weaviate)
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
    logger.info("Pipeline complete  [strategy=%s].", chunking_strategy)
    logger.info("=" * 60)

    return {
        "documents_parsed": len(documents),
        "chunks_created":   len(chunks),
        "chunks_stored":    stored,
        "total_in_db":      total_in_db,
    }
