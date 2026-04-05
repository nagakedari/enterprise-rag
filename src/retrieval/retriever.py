"""
Hybrid + semantic retrieval from Weaviate.

Retrieval modes (config.retrieval.mode / RETRIEVAL_MODE env var):

    "semantic"  (default)
        Pure cosine-similarity search via near_vector.
        Best for conceptual / paraphrase queries.

    "hybrid"
        BM25 (lexical) + cosine (semantic) fused via Reciprocal Rank Fusion.
        config.retrieval.alpha (RETRIEVAL_HYBRID_ALPHA) controls the balance:
            0.0 = pure BM25   – exact keyword matching
            0.5 = equal weight (default)
            1.0 = pure vector  – same as semantic mode
        Best for queries with exact financial terms, ticker symbols, or
        specific numbers that benefit from keyword matching.

BM25 targets only content fields (chunk_text / parent_text + section_title).
Metadata fields (company, quarter, source_file …) are excluded from BM25
to avoid spurious matches.

Supports two collections (collection_name parameter):
    SecDocument       – Phase-1 basic chunking; returns chunk_text
    SecDocumentSmart  – Phase-2 parent-child; returns parent_text,
                        deduplicates by parent_id

Usage:
    # Semantic, basic collection
    chunks = retrieve("Apple revenue", config)

    # Hybrid, smart collection
    chunks = retrieve("Apple revenue", config,
                      collection_name=config.weaviate.smart_collection_name)

    # Override mode / alpha per call
    chunks = retrieve("Apple revenue", config, mode="hybrid", alpha=0.3)
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import weaviate
from weaviate.classes.query import MetadataQuery, Filter

from src.config import Config
from src.ingestion.embedder import create_embeddings
from src.ingestion.weaviate_store import get_client

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval with a comparable relevance score."""
    text: str           # content returned to the LLM
                        #   SecDocument      → chunk_text
                        #   SecDocumentSmart → parent_text (deduplicated)
    company: str
    quarter: str
    year: int
    source_file: str
    chunk_index: int
    page_start: int
    page_end: int
    token_count: int
    distance: float     # cosine distance  (semantic mode, lower = better)
                        # set to 0.0 for hybrid mode
    search_score: float = field(default=0.0)
                        # RRF fusion score  (hybrid mode, higher = better)
                        # set to 0.0 for semantic mode
    section_title: str = field(default="")

    @property
    def score(self) -> float:
        """
        Unified relevance score (higher = better) regardless of mode.
          semantic → 1.0 - cosine_distance
          hybrid   → RRF fusion score (relative; not normalised to [0,1])
        """
        if self.search_score > 0.0:
            return self.search_score
        return 1.0 - self.distance


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_filters(
    company: Optional[str],
    year: Optional[int],
    quarter: Optional[str],
) -> Optional[Filter]:
    """AND-combine any provided metadata filters."""
    parts: List[Filter] = []
    if company:
        parts.append(Filter.by_property("company").equal(company))
    if year is not None:
        parts.append(Filter.by_property("year").equal(year))
    if quarter:
        parts.append(Filter.by_property("quarter").equal(quarter))

    if not parts:
        return None
    result = parts[0]
    for f in parts[1:]:
        result = result & f
    return result


def _map_object(obj, is_smart: bool, distance: float, search_score: float) -> RetrievedChunk:
    """Map a raw Weaviate result object to a RetrievedChunk."""
    p = obj.properties
    return RetrievedChunk(
        text=(p.get("parent_text", "") if is_smart else p.get("chunk_text", "")),
        company=p.get("company", ""),
        quarter=p.get("quarter", ""),
        year=int(p.get("year", 0)),
        source_file=(
            p.get("file_name", "") or p.get("source_file", "")
            if is_smart else p.get("source_file", "")
        ),
        chunk_index=int(p.get("chunk_index", 0)),
        page_start=int(p.get("page_num", 0) if is_smart else p.get("page_start", 0)),
        page_end=int(p.get("page_num", 0) if is_smart else p.get("page_end", 0)),
        token_count=int(p.get("token_count", 0)),
        distance=distance,
        search_score=search_score,
        section_title=p.get("section_title", "") if is_smart else "",
    )


def _deduplicate(
    results,
    is_smart: bool,
    top_k: int,
    mode: str,
) -> List[RetrievedChunk]:
    """
    Map Weaviate results to RetrievedChunks.
    For the smart collection, deduplicate by parent_id so each parent
    appears at most once, using the best-scoring child hit per parent.
    """
    chunks: List[RetrievedChunk] = []
    seen_parent_ids: set = set()

    for obj in results.objects:
        p = obj.properties
        meta = obj.metadata

        if mode == "hybrid":
            distance = 0.0
            search_score = meta.score if meta.score is not None else 0.0
        else:
            distance = meta.distance if meta.distance is not None else 1.0
            search_score = 0.0

        if is_smart:
            parent_id = p.get("parent_id", "")
            if parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(parent_id)

        chunks.append(_map_object(obj, is_smart, distance, search_score))

        if len(chunks) == top_k:
            break

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    config: Config,
    top_k: int = 5,
    company: Optional[str] = None,
    year: Optional[int] = None,
    quarter: Optional[str] = None,
    collection_name: Optional[str] = None,
    mode: Optional[str] = None,
    alpha: Optional[float] = None,
) -> List[RetrievedChunk]:
    """
    Retrieve the top-k most relevant chunks for *query*.

    Args:
        query:           Natural-language question.
        config:          Project-wide Config.
        top_k:           Maximum results to return.
        company:         Filter by ticker (e.g. "AAPL").
        year:            Filter by year  (e.g. 2023).
        quarter:         Filter by quarter (e.g. "Q2").
        collection_name: Weaviate collection to query.
                         Defaults to config.weaviate.collection_name.
        mode:            "semantic" or "hybrid".  Defaults to
                         config.retrieval.mode (env: RETRIEVAL_MODE).
        alpha:           BM25/vector balance for hybrid mode (0–1).
                         Defaults to config.retrieval.alpha
                         (env: RETRIEVAL_HYBRID_ALPHA).

    Returns:
        List of RetrievedChunk ordered by relevance (best first).
    """
    target_collection = collection_name or config.weaviate.collection_name
    is_smart = target_collection == config.weaviate.smart_collection_name
    effective_mode = mode or config.retrieval.mode
    effective_alpha = alpha if alpha is not None else config.retrieval.alpha

    logger.info(
        "retrieve  mode=%s  alpha=%.2f  collection=%s  top_k=%d  "
        "query=%r  filters=(company=%s year=%s quarter=%s)",
        effective_mode, effective_alpha, target_collection, top_k,
        query, company, year, quarter,
    )

    filters = _build_filters(company, year, quarter)

    # Fetch more candidates for smart collection so dedup still yields top_k
    fetch_limit = top_k * 3 if is_smart else top_k

    # BM25 searches only the content property (not metadata strings)
    query_properties = (
        ["parent_text", "section_title"] if is_smart else ["chunk_text"]
    )

    client: weaviate.WeaviateClient = get_client(config.weaviate)
    try:
        collection = client.collections.get(target_collection)

        if effective_mode == "hybrid":
            # Embed query for the vector half of hybrid search
            query_vector = create_embeddings([query], config.embedding)[0]
            results = collection.query.hybrid(
                query=query,                      # BM25 leg
                vector=query_vector,              # semantic leg
                alpha=effective_alpha,
                query_properties=query_properties,
                limit=fetch_limit,
                filters=filters,
                return_metadata=MetadataQuery(score=True),
            )
        else:
            # Pure semantic
            query_vector = create_embeddings([query], config.embedding)[0]
            results = collection.query.near_vector(
                near_vector=query_vector,
                limit=fetch_limit,
                filters=filters,
                return_metadata=MetadataQuery(distance=True),
            )
    finally:
        client.close()

    chunks = _deduplicate(results, is_smart, top_k, effective_mode)

    if chunks:
        logger.info(
            "Retrieved %d chunks  top_score=%.4f  mode=%s",
            len(chunks), chunks[0].score, effective_mode,
        )
    return chunks
