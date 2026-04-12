"""
LlamaIndex-based retriever for SEC 10-Q filings.

Replaces custom retriever.py Weaviate client calls with LlamaIndex abstractions:

    VectorStoreIndex.as_retriever()  → semantic (near_vector)
    VectorStoreQueryMode.HYBRID      → BM25 + vector (Weaviate hybrid)
    AutoMergingRetriever             → smart mode: returns parent node text
                                       when enough child nodes from the same
                                       parent are retrieved

Returns the same ``RetrievedChunk`` dataclass as the custom retriever so the
generator and evaluator need no changes.

Mode / alpha are read from config.retrieval (env: RETRIEVAL_MODE /
RETRIEVAL_HYBRID_ALPHA) and can be overridden per call.
"""
import logging
from pathlib import Path
from typing import List, Optional

from src.config import Config
from src.ingestion.weaviate_store import get_client
from src.retrieval.retriever import RetrievedChunk   # reuse existing dataclass

logger = logging.getLogger(__name__)


def retrieve_llamaindex(
    query: str,
    config: Config,
    top_k: int = 5,
    company: Optional[str] = None,
    year: Optional[int] = None,
    quarter: Optional[str] = None,
    use_smart: bool = False,
    mode: Optional[str] = None,
    alpha: Optional[float] = None,
) -> List[RetrievedChunk]:
    """
    Retrieve the top-k most relevant chunks using LlamaIndex + Weaviate.

    Args:
        query:     Natural-language question.
        config:    Project-wide Config.
        top_k:     Maximum results to return.
        company:   Filter by ticker (e.g. "AAPL").
        year:      Filter by year   (e.g. 2023).
        quarter:   Filter by quarter (e.g. "Q2").
        use_smart: False → query SecDocumentLI  (basic nodes)
                   True  → query SecDocumentSmartLI + AutoMergingRetriever
                           (child vector match → return parent text)
        mode:      "semantic" or "hybrid".  Defaults to config.retrieval.mode.
        alpha:     BM25/vector balance for hybrid (0–1).
                   Defaults to config.retrieval.alpha.

    Returns:
        List of RetrievedChunk ordered by relevance (best first).
        Same type as custom retrieve() so generator / evaluator are unchanged.
    """
    # ── lazy imports ───────────────────────────────────────────────────────────
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.vector_stores.types import (
        MetadataFilter,
        MetadataFilters,
        FilterOperator,
        VectorStoreQueryMode,
    )
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.vector_stores.weaviate import WeaviateVectorStore

    effective_mode = mode or config.retrieval.mode
    effective_alpha = alpha if alpha is not None else config.retrieval.alpha
    collection_name = (
        config.weaviate.llamaindex_smart_collection_name
        if use_smart
        else config.weaviate.llamaindex_collection_name
    )

    logger.info(
        "retrieve_llamaindex  mode=%s  alpha=%.2f  collection=%s  "
        "top_k=%d  query=%r  filters=(company=%s year=%s quarter=%s)",
        effective_mode, effective_alpha, collection_name,
        top_k, query, company, year, quarter,
    )

    # ── Metadata filters ───────────────────────────────────────────────────────
    filter_list = []
    if company:
        filter_list.append(
            MetadataFilter(key="company", value=company, operator=FilterOperator.EQ)
        )
    if year is not None:
        filter_list.append(
            MetadataFilter(key="year", value=year, operator=FilterOperator.EQ)
        )
    if quarter:
        filter_list.append(
            MetadataFilter(key="quarter", value=quarter, operator=FilterOperator.EQ)
        )
    filters = MetadataFilters(filters=filter_list) if filter_list else None

    # ── Connect to Weaviate ────────────────────────────────────────────────────
    weaviate_client = get_client(config.weaviate)
    embed_model = OpenAIEmbedding(
        model=config.embedding.model,
        api_key=config.embedding.api_key,
    )

    try:
        vector_store = WeaviateVectorStore(
            weaviate_client=weaviate_client,
            index_name=collection_name,
        )
        # llama-index-vector-stores-weaviate 1.1.x has a Pydantic v2 bug where
        # _client (declared as PrivateAttr) is never populated in __init__.
        # Inject it directly into __pydantic_private__ to avoid AttributeError
        # deep inside WeaviateVectorStore.query → get_all_properties(self._client).
        _priv = getattr(vector_store, "__pydantic_private__", None)
        if isinstance(_priv, dict) and "_client" not in _priv:
            _priv["_client"] = weaviate_client

        if use_smart:
            # Load persisted docstore so AutoMergingRetriever can fetch parents
            from llama_index.core.retrievers import AutoMergingRetriever

            persist_dir = Path(config.llamaindex_docstore_path)
            docstore_file = persist_dir / "docstore.json"
            if not docstore_file.exists():
                raise RuntimeError(
                    f"LlamaIndex smart docstore not found at '{docstore_file}'. "
                    "Re-run ingestion with engine='llamaindex' and use_smart=True "
                    "to build and persist the parent-node docstore before using "
                    "smart retrieval."
                )
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=str(persist_dir),
            )
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=embed_model,
            )
            # Fetch extra candidates before merging (same pattern as custom retriever)
            base_retriever = index.as_retriever(
                similarity_top_k=top_k * 3,
                vector_store_query_mode=(
                    VectorStoreQueryMode.HYBRID
                    if effective_mode == "hybrid"
                    else VectorStoreQueryMode.DEFAULT
                ),
                alpha=effective_alpha if effective_mode == "hybrid" else None,
                filters=filters,
            )
            retriever = AutoMergingRetriever(
                base_retriever,
                storage_context=storage_context,
                verbose=False,
            )
        else:
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=embed_model,
            )
            retriever = index.as_retriever(
                similarity_top_k=top_k,
                vector_store_query_mode=(
                    VectorStoreQueryMode.HYBRID
                    if effective_mode == "hybrid"
                    else VectorStoreQueryMode.DEFAULT
                ),
                alpha=effective_alpha if effective_mode == "hybrid" else None,
                filters=filters,
            )

        node_results = retriever.retrieve(query)

    finally:
        weaviate_client.close()

    # ── Map NodeWithScore → RetrievedChunk ─────────────────────────────────────
    chunks: List[RetrievedChunk] = []
    for node_with_score in node_results[:top_k]:
        node = node_with_score.node
        meta = node.metadata or {}
        score = node_with_score.score or 0.0

        # LlamaIndex hybrid returns a positive fusion score (higher = better).
        # Semantic returns a similarity score in [0, 1].
        # Map to distance (lower = better) for consistency with RetrievedChunk.
        distance = max(0.0, 1.0 - score) if effective_mode == "semantic" else 0.0
        search_score = score if effective_mode == "hybrid" else 0.0

        chunks.append(RetrievedChunk(
            text=node.get_content(),
            company=str(meta.get("company", "")),
            quarter=str(meta.get("quarter", "")),
            year=int(meta.get("year", 0) or 0),
            source_file=str(meta.get("file_name", meta.get("file_path", ""))),
            chunk_index=0,          # LlamaIndex uses node_id instead of sequential index
            page_start=int(meta.get("page_label", 0) or 0),
            page_end=int(meta.get("page_label", 0) or 0),
            token_count=0,          # not stored by LlamaIndex by default
            distance=distance,
            search_score=search_score,
            section_title=str(meta.get("section_title", "")),
        ))

    logger.info(
        "retrieve_llamaindex returned %d chunks  top_score=%.4f  mode=%s",
        len(chunks), chunks[0].score if chunks else 0.0, effective_mode,
    )
    return chunks
