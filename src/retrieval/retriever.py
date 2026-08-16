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

BM25 targets only content fields — metadata fields are excluded to avoid
spurious matches.  The content field differs by collection:
    SecDocument      → "chunk_text"
    SecDocumentSmart → "parent_text", "section_title"
    DocumentChunk    → "text_for_search"  (title-prefixed + overlap)

Supports three collections (collection_name parameter):
    SecDocument       – Phase-1 basic chunking; returns chunk_text
    SecDocumentSmart  – Phase-2 parent-child; returns parent_text,
                        deduplicates by parent_id
    DocumentChunk     – Phase-3 semantic chunking; returns raw_content,
                        embeds query with BGE-M3 (same model used at ingest)

Usage:
    # Semantic mode, basic collection
    chunks = retrieve("Apple revenue", config)

    # Hybrid mode, parent-child collection
    chunks = retrieve("Apple revenue", config,
                      collection_name=config.weaviate.smart_collection_name)

    # Hybrid mode, semantic collection (uses BGE-M3 for query embedding)
    chunks = retrieve("Apple revenue", config,
                      collection_name=config.weaviate.semantic_collection_name)
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


def _map_object(
    obj,
    is_smart: bool,
    is_semantic: bool,
    distance: float,
    search_score: float,
) -> RetrievedChunk:
    """Map a raw Weaviate result object to a RetrievedChunk."""
    p = obj.properties
    if is_semantic:
        text = p.get("raw_content", "")
        source_file = p.get("file_name", "") or p.get("source_file", "")
        page = int(p.get("page_num", 0))
        section_title = p.get("section_title", "")
    elif is_smart:
        text = p.get("parent_text", "")
        source_file = p.get("file_name", "") or p.get("source_file", "")
        page = int(p.get("page_num", 0))
        section_title = p.get("section_title", "")
    else:
        text = p.get("chunk_text", "")
        source_file = p.get("source_file", "")
        page = 0
        section_title = ""

    return RetrievedChunk(
        text=text,
        company=p.get("company", ""),
        quarter=p.get("quarter", ""),
        year=int(p.get("year", 0)),
        source_file=source_file,
        chunk_index=int(p.get("chunk_index", 0)),
        page_start=page if (is_smart or is_semantic) else int(p.get("page_start", 0)),
        page_end=page if (is_smart or is_semantic) else int(p.get("page_end", 0)),
        token_count=int(p.get("token_count", 0)),
        distance=distance,
        search_score=search_score,
        section_title=section_title,
    )


def _deduplicate(
    results,
    is_smart: bool,
    is_semantic: bool,
    top_k: int,
    mode: str,
) -> List[RetrievedChunk]:
    """
    Map Weaviate results to RetrievedChunks.
    For the parent-child collection, deduplicate by parent_id so each parent
    appears at most once, using the best-scoring child hit per parent.
    Semantic and basic collections have no deduplication step.
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

        chunks.append(_map_object(obj, is_smart, is_semantic, distance, search_score))

        if len(chunks) == top_k:
            break

    return chunks


# ── Per-filing retrieval helpers ──────────────────────────────────────────────

def _discover_filings(
    collection,
    company: str,
) -> List[tuple]:
    """
    Return the distinct (year, quarter) pairs stored for *company*.

    Uses fetch_objects with a company filter to avoid a full table scan.
    The limit of 500 is safely above the number of filings we expect
    (quarterly filings since ~2020 = ~20 per company).
    """
    company_filter = Filter.by_property("company").equal(company)
    results = collection.query.fetch_objects(
        filters=company_filter,
        limit=500,
        return_properties=["year", "quarter"],
    )
    seen: set = set()
    filings: List[tuple] = []
    for obj in results.objects:
        p = obj.properties
        key = (int(p.get("year", 0)), p.get("quarter", ""))
        if key not in seen and key[0] > 0:
            seen.add(key)
            filings.append(key)
    filings.sort()
    logger.info("Discovered %d filings for %s: %s", len(filings), company, filings)
    return filings


def retrieve_per_filing(
    query: str,
    config: Config,
    company: str,
    chunks_per_filing: int = 3,
    collection_name: Optional[str] = None,
    mode: Optional[str] = None,
    alpha: Optional[float] = None,
) -> List[RetrievedChunk]:
    """
    Run one retrieval query per (year, quarter) filing for *company*, then
    merge and deduplicate the results.

    Use this when the question does not specify a quarter and the answer
    requires facts from multiple filings (e.g. "How has Apple's net sales
    changed over time?").  A single global query with quarter=null tends to
    return N chunks from one filing; per-filing retrieval guarantees temporal
    coverage.

    Args:
        query:              Natural-language question.
        config:             Project-wide Config.
        company:            Ticker symbol (e.g. "AAPL").
        chunks_per_filing:  How many chunks to fetch per (year, quarter) pair.
        collection_name:    Weaviate collection override.
        mode:               "semantic" or "hybrid".
        alpha:              Hybrid alpha (0–1).

    Returns:
        Merged, deduplicated list of RetrievedChunks, best-first within each
        filing.  Ordering across filings follows (year, quarter) ascending
        so temporal ordering is preserved for the reranker.
    """
    target_collection = collection_name or config.weaviate.collection_name
    is_smart = target_collection == config.weaviate.smart_collection_name
    is_semantic = target_collection == config.weaviate.semantic_collection_name
    effective_mode = mode or config.retrieval.mode
    effective_alpha = alpha if alpha is not None else config.retrieval.alpha

    if is_semantic:
        from src.ingestion.semantic_chunker import create_bge_embeddings
        query_vector = create_bge_embeddings(
            [query],
            config.semantic_chunk.embedding_model,
            config.semantic_chunk.use_onnx,
        )[0]
    else:
        query_vector = create_embeddings([query], config.embedding)[0]

    if is_semantic:
        query_properties = ["text_for_search"]
    elif is_smart:
        query_properties = ["parent_text", "section_title"]
    else:
        query_properties = ["chunk_text"]

    client: weaviate.WeaviateClient = get_client(config.weaviate)
    try:
        collection = client.collections.get(target_collection)
        filings = _discover_filings(collection, company)

        if not filings:
            logger.warning("No filings found for %s — falling back to global retrieve.", company)
            client.close()
            return retrieve(query, config, top_k=chunks_per_filing * 4,
                            company=company, collection_name=collection_name,
                            mode=mode, alpha=alpha)

        all_chunks: List[RetrievedChunk] = []
        seen_dedup: set = set()

        for year, quarter in filings:
            f_filter = _build_filters(company, year, quarter)
            fetch_limit = chunks_per_filing * 3 if is_smart else chunks_per_filing
            try:
                if effective_mode == "hybrid":
                    results = collection.query.hybrid(
                        query=query,
                        vector=query_vector,
                        alpha=effective_alpha,
                        query_properties=query_properties,
                        limit=fetch_limit,
                        filters=f_filter,
                        return_metadata=MetadataQuery(score=True),
                    )
                else:
                    results = collection.query.near_vector(
                        near_vector=query_vector,
                        limit=fetch_limit,
                        filters=f_filter,
                        return_metadata=MetadataQuery(distance=True),
                    )
            except Exception as exc:
                logger.warning("Per-filing query failed for %s %s %d: %s", company, quarter, year, exc)
                continue

            filing_chunks = _deduplicate(results, is_smart, is_semantic, chunks_per_filing, effective_mode)
            for chunk in filing_chunks:
                dedup_key = (chunk.source_file, chunk.chunk_index)
                if dedup_key not in seen_dedup:
                    seen_dedup.add(dedup_key)
                    all_chunks.append(chunk)

    finally:
        client.close()

    logger.info(
        "Per-filing retrieval: %d filings × %d chunks → %d unique chunks  company=%s",
        len(filings), chunks_per_filing, len(all_chunks), company,
    )
    return all_chunks


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
    is_semantic = target_collection == config.weaviate.semantic_collection_name
    effective_mode = mode or config.retrieval.mode
    effective_alpha = alpha if alpha is not None else config.retrieval.alpha

    logger.info(
        "retrieve  mode=%s  alpha=%.2f  collection=%s  top_k=%d  "
        "query=%r  filters=(company=%s year=%s quarter=%s)",
        effective_mode, effective_alpha, target_collection, top_k,
        query, company, year, quarter,
    )

    filters = _build_filters(company, year, quarter)

    # Fetch extra candidates for smart collection to compensate for parent dedup
    fetch_limit = top_k * 3 if is_smart else top_k

    # BM25 searches only the dedicated content field for each collection type
    if is_semantic:
        query_properties = ["text_for_search"]
    elif is_smart:
        query_properties = ["parent_text", "section_title"]
    else:
        query_properties = ["chunk_text"]

    # Semantic (DocumentChunk) collection uses BGE-M3 for query embedding to match
    # the model used at ingest time.  All other collections use OpenAI.
    if is_semantic:
        from src.ingestion.semantic_chunker import create_bge_embeddings
        query_vector = create_bge_embeddings(
            [query],
            config.semantic_chunk.embedding_model,
            config.semantic_chunk.use_onnx,
        )[0]
    else:
        query_vector = create_embeddings([query], config.embedding)[0]

    client: weaviate.WeaviateClient = get_client(config.weaviate)
    try:
        collection = client.collections.get(target_collection)

        if effective_mode == "hybrid":
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
            # Pure vector search
            results = collection.query.near_vector(
                near_vector=query_vector,
                limit=fetch_limit,
                filters=filters,
                return_metadata=MetadataQuery(distance=True),
            )
    finally:
        client.close()

    chunks = _deduplicate(results, is_smart, is_semantic, top_k, effective_mode)

    if chunks:
        logger.info(
            "Retrieved %d chunks  top_score=%.4f  mode=%s",
            len(chunks), chunks[0].score, effective_mode,
        )
    return chunks
