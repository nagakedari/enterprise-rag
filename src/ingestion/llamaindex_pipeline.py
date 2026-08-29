"""
LlamaIndex-based ingestion pipeline for SEC 10-Q filings.

Replaces custom pdf_parser + chunker + embedder + weaviate_store with
LlamaIndex abstractions:

    SimpleDirectoryReader      → PDF loading with automatic metadata extraction
    SentenceSplitter           → basic mode  (~800-token nodes)
    HierarchicalNodeParser     → smart mode  (parent ~1000t → child ~300t nodes)
    OpenAIEmbedding            → embedding (same model as custom pipeline)
    WeaviateVectorStore        → Weaviate backend (LI-specific collections)
    VectorStoreIndex           → orchestrates embed + store in one call

Smart mode additionally persists a docstore to disk so that
AutoMergingRetriever (in llamaindex_retriever.py) can look up parent nodes
at query time without requiring a second Weaviate collection.

Collections used (separate from custom pipeline to allow A/B comparison):
    basic  → config.weaviate.llamaindex_collection_name   (SecDocumentLI)
    smart  → config.weaviate.llamaindex_smart_collection_name  (SecDocumentSmartLI)
"""
import logging
import re
from pathlib import Path
from typing import Optional

import tiktoken

from src.config import Config
from src.ingestion.weaviate_store import get_client

logger = logging.getLogger(__name__)

# ── Metadata extraction ────────────────────────────────────────────────────────

_FILENAME_RE = re.compile(r"^(\d{4})\s+(Q[1-4])\s+([A-Z]+)$", re.IGNORECASE)


def _file_metadata(filepath: str) -> dict:
    """
    Extract company / quarter / year from the SEC PDF filename convention
    "YYYY QX TICKER.pdf" and return as a LlamaIndex metadata dict.
    LlamaIndex merges this with its own auto-extracted keys (file_name,
    file_path, creation_date, last_modified_date, page_label).
    """
    stem = Path(filepath).stem          # "2023 Q2 AAPL"
    name = Path(filepath).name          # "2023 Q2 AAPL.pdf"
    m = _FILENAME_RE.match(stem.strip())
    if m:
        return {
            "year":     int(m.group(1)),
            "quarter":  m.group(2).upper(),
            "company":  m.group(3).upper(),
            "file_name": name,
        }
    logger.warning("Could not parse metadata from filename: %s", name)
    return {"file_name": name}


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_llamaindex_pipeline(
    docs_path: Optional[Path] = None,
    config: Optional[Config] = None,
    use_smart: bool = False,
    recreate_collection: bool = False,
) -> dict:
    """
    Run the LlamaIndex ingestion pipeline.

    Args:
        docs_path:           Override docs directory from config.
        config:              Project-wide Config; uses env defaults if None.
        use_smart:           False → SentenceSplitter → SecDocumentLI
                             True  → HierarchicalNodeParser → SecDocumentSmartLI
                                     + persists docstore for AutoMergingRetriever
        recreate_collection: Drop the target Weaviate collection before ingest.

    Returns:
        Summary dict::

            {
                "documents_loaded":  int,
                "nodes_indexed":     int,
                "collection":        str,
            }
    """
    # ── lazy imports keep startup fast when LlamaIndex is not installed ────────
    from llama_index.core import (
        SimpleDirectoryReader,
        StorageContext,
        VectorStoreIndex,
    )
    from llama_index.core.node_parser import (
        HierarchicalNodeParser,
        SentenceSplitter,
        get_leaf_nodes,
    )
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.vector_stores.weaviate import WeaviateVectorStore

    if config is None:
        config = Config()
    if docs_path is not None:
        config.docs_path = Path(docs_path)

    collection_name = (
        config.weaviate.llamaindex_smart_collection_name
        if use_smart
        else config.weaviate.llamaindex_collection_name
    )
    mode = "smart (HierarchicalNodeParser)" if use_smart else "basic (SentenceSplitter)"

    logger.info("=" * 60)
    logger.info("Starting LlamaIndex ingestion pipeline  [mode=%s]", mode)
    logger.info("  docs_path  : %s", config.docs_path)
    logger.info("  collection : %s", collection_name)
    logger.info("  embedding  : %s", config.embedding.model)
    logger.info("=" * 60)

    # ── Step 1: Load PDFs ──────────────────────────────────────────────────────
    logger.info("[1/4] Loading PDFs with SimpleDirectoryReader …")
    reader = SimpleDirectoryReader(
        input_dir=str(config.docs_path),
        required_exts=[".pdf"],
        file_metadata=_file_metadata,
    )
    documents = reader.load_data()
    logger.info("      → %d document pages loaded", len(documents))

    # ── Step 2: Parse into nodes ───────────────────────────────────────────────
    logger.info("[2/4] Parsing nodes  [mode=%s] …", mode)
    enc = tiktoken.get_encoding(config.chunk.encoding)

    if use_smart:
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[
                config.smart_chunk.parent_max_tokens,
                config.smart_chunk.child_max_tokens,
            ],
        )
        all_nodes = node_parser.get_nodes_from_documents(documents)
        nodes_to_index = get_leaf_nodes(all_nodes)   # only leaf (child) nodes go into the vector store
    else:
        node_parser = SentenceSplitter(
            chunk_size=config.chunk.max_tokens,
            chunk_overlap=config.chunk.overlap_tokens,
            tokenizer=enc.encode,
        )
        nodes_to_index = node_parser.get_nodes_from_documents(documents)
        all_nodes = nodes_to_index

    logger.info(
        "      → %d total nodes  (%d to index)",
        len(all_nodes), len(nodes_to_index),
    )

    # ── Step 3: Connect to Weaviate and optionally recreate collection ─────────
    logger.info("[3/4] Connecting to Weaviate …")
    weaviate_client = get_client(config.weaviate)

    try:
        if recreate_collection and weaviate_client.collections.exists(collection_name):
            weaviate_client.collections.delete(collection_name)
            logger.info("      Dropped collection '%s'.", collection_name)

        embed_model = OpenAIEmbedding(
            model=config.embedding.model,
            api_key=config.embedding.api_key,
        )

        vector_store = WeaviateVectorStore(
            weaviate_client=weaviate_client,
            index_name=collection_name,
        )
        # llama-index-vector-stores-weaviate 1.1.x Pydantic v2 bug: _client PrivateAttr
        # is never set in __init__. Inject it directly so embed/store calls don't fail.
        _priv = getattr(vector_store, "__pydantic_private__", None)
        if isinstance(_priv, dict) and "_client" not in _priv:
            _priv["_client"] = weaviate_client

        if use_smart:
            # Persist all nodes (parents + children) so AutoMergingRetriever
            # can fetch parent text at query time without a second DB round-trip.
            from llama_index.core.storage.docstore import SimpleDocumentStore

            docstore = SimpleDocumentStore()
            docstore.add_documents(all_nodes)

            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                docstore=docstore,
            )
        else:
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # ── Step 4: Build index (embeds + stores in one call) ──────────────────
        logger.info("[4/4] Embedding and storing via VectorStoreIndex …")
        VectorStoreIndex(
            nodes=nodes_to_index,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )

        if use_smart:
            # Persist docstore to disk for use by the retriever.
            # mkdir ensures the path exists even if config points to a new location.
            persist_dir = Path(config.llamaindex_docstore_path)
            persist_dir.mkdir(parents=True, exist_ok=True)
            storage_context.persist(persist_dir=str(persist_dir))
            logger.info("      Docstore persisted to %s", persist_dir)

    finally:
        weaviate_client.close()

    logger.info("=" * 60)
    logger.info("LlamaIndex pipeline complete  [mode=%s].", mode)
    logger.info("=" * 60)

    return {
        "documents_loaded": len(documents),
        "nodes_indexed":    len(nodes_to_index),
        "collection":       collection_name,
    }
