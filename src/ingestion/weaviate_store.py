"""
Weaviate v4 client wrapper.

Schema:
    Collection  : SecDocument
    Vectorizer  : none  (we push pre-computed OpenAI vectors)
    Distance    : cosine

Properties stored per chunk:
    company       – ticker symbol  (AAPL, AMZN, …)
    quarter       – Q1 / Q2 / Q3 / Q4
    year          – 2022 / 2023
    source_file   – original PDF filename
    chunk_index   – sequential index within the document
    chunk_text    – the raw text of the chunk
    page_start    – first PDF page the chunk spans
    page_end      – last PDF page the chunk spans
    token_count   – number of tokens in the chunk

Phase-1 note:
    UUIDs are deterministic (uuid5) so re-running ingestion is idempotent –
    existing objects are silently skipped / overwritten by Weaviate.
"""
import logging
import uuid
from typing import List

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances, Tokenization

from src.config import WeaviateConfig
from src.ingestion.chunker import TextChunk
from src.ingestion.smart_chunker import SmartTextChunk
from src.ingestion.semantic_chunker import SemanticTextChunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "SecDocument"


# ── Connection ────────────────────────────────────────────────────────────────

def get_client(config: WeaviateConfig) -> weaviate.WeaviateClient:
    """Return an open Weaviate client.  Caller is responsible for .close()."""
    client = weaviate.connect_to_custom(
        http_host=config.http_host,
        http_port=config.http_port,
        http_secure=False,
        grpc_host=config.http_host,
        grpc_port=config.grpc_port,
        grpc_secure=False,
    )
    logger.info(
        "Connected to Weaviate at %s:%d", config.http_host, config.http_port
    )
    return client


# ── Schema management ─────────────────────────────────────────────────────────

def ensure_schema(client: weaviate.WeaviateClient, config: WeaviateConfig) -> None:
    """Create the SecDocument collection if it does not already exist."""
    if client.collections.exists(config.collection_name):
        logger.info("Collection '%s' already exists – skipping creation.", config.collection_name)
        return

    client.collections.create(
        name=config.collection_name,
        description="SEC 10-Q filing document chunks with OpenAI embeddings",
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
        ),
        properties=[
            # Metadata fields – filterable but not BM25-searched
            Property(name="company",     data_type=DataType.TEXT, index_searchable=False),
            Property(name="quarter",     data_type=DataType.TEXT, index_searchable=False),
            Property(name="year",        data_type=DataType.INT),
            Property(name="source_file", data_type=DataType.TEXT, index_searchable=False),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="page_start",  data_type=DataType.INT),
            Property(name="page_end",    data_type=DataType.INT),
            Property(name="token_count", data_type=DataType.INT),
            # Content field – BM25-indexed for hybrid search
            Property(
                name="chunk_text",
                data_type=DataType.TEXT,
                index_searchable=True,
                tokenization=Tokenization.WORD,
            ),
        ],
    )
    logger.info("Created collection '%s'.", config.collection_name)


def drop_collection(client: weaviate.WeaviateClient, config: WeaviateConfig) -> None:
    """Drop and recreate the collection (useful for a clean re-ingest)."""
    if client.collections.exists(config.collection_name):
        client.collections.delete(config.collection_name)
        logger.info("Dropped collection '%s'.", config.collection_name)


# ── Data ingestion ────────────────────────────────────────────────────────────

def _chunk_uuid(chunk: TextChunk) -> str:
    """Deterministic UUID based on source file + chunk index."""
    return str(
        uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.source_file}::{chunk.chunk_index}")
    )


def store_chunks(
    client: weaviate.WeaviateClient,
    chunks: List[TextChunk],
    embeddings: List[List[float]],
    config: WeaviateConfig,
) -> int:
    """
    Batch-insert chunks with their embedding vectors.
    Returns the number of objects successfully queued for insertion.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
        )

    collection = client.collections.get(config.collection_name)
    stored = 0

    with collection.batch.fixed_size(batch_size=config.batch_size) as batch:
        for chunk, vector in zip(chunks, embeddings):
            batch.add_object(
                properties={
                    "company":     chunk.company,
                    "quarter":     chunk.quarter,
                    "year":        chunk.year,
                    "source_file": chunk.source_file,
                    "chunk_index": chunk.chunk_index,
                    "chunk_text":  chunk.text,
                    "page_start":  chunk.page_start,
                    "page_end":    chunk.page_end,
                    "token_count": chunk.token_count,
                },
                vector=vector,
                uuid=_chunk_uuid(chunk),
            )
            stored += 1

    logger.info("Queued %d objects for insertion into '%s'.", stored, config.collection_name)
    return stored


# ── Verification ──────────────────────────────────────────────────────────────

def get_total_count(client: weaviate.WeaviateClient, config: WeaviateConfig) -> int:
    """Return total number of objects in the SecDocument collection."""
    collection = client.collections.get(config.collection_name)
    result = collection.aggregate.over_all(total_count=True)
    return result.total_count


# ── SecDocumentSmart – schema, ingestion, verification ────────────────────────

def ensure_schema_smart(client: weaviate.WeaviateClient, config: WeaviateConfig) -> None:
    """
    Create the SecDocumentSmart collection if it does not already exist.

    Schema differences from SecDocument:
        chunk_text    – child chunk text (~300 tokens); this is what is vectorized
        parent_text   – parent chunk text (~1000 tokens); returned to the LLM
        parent_id     – UUID string; used by the retriever to deduplicate results
        section_title – nearest preceding section header in the parent
    """
    if client.collections.exists(config.smart_collection_name):
        logger.info(
            "Collection '%s' already exists – skipping creation.",
            config.smart_collection_name,
        )
        return

    client.collections.create(
        name=config.smart_collection_name,
        description=(
            "SEC 10-Q parent-child chunks. "
            "Vector = child context_window embedding; parent_text returned to LLM."
        ),
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
        ),
        properties=[
            # Metadata fields – filterable but not BM25-searched
            Property(name="company",           data_type=DataType.TEXT, index_searchable=False),
            Property(name="quarter",           data_type=DataType.TEXT, index_searchable=False),
            Property(name="year",              data_type=DataType.INT),
            Property(name="file_name",         data_type=DataType.TEXT, index_searchable=False),
            Property(name="source_file",       data_type=DataType.TEXT, index_searchable=False),
            Property(name="chunk_index",       data_type=DataType.INT),
            Property(name="parent_id",         data_type=DataType.TEXT, index_searchable=False),
            Property(name="page_num",          data_type=DataType.INT),
            Property(name="token_count",       data_type=DataType.INT),
            Property(name="last_updated_date", data_type=DataType.TEXT, index_searchable=False),
            # Content fields – BM25-indexed for hybrid search
            Property(
                name="parent_text",
                data_type=DataType.TEXT,
                index_searchable=True,
                tokenization=Tokenization.WORD,
            ),
            Property(
                name="section_title",
                data_type=DataType.TEXT,
                index_searchable=True,
                tokenization=Tokenization.WORD,
            ),
        ],
    )
    logger.info("Created collection '%s'.", config.smart_collection_name)


def drop_collection_smart(client: weaviate.WeaviateClient, config: WeaviateConfig) -> None:
    """Drop the SecDocumentSmart collection (useful for clean re-ingest)."""
    if client.collections.exists(config.smart_collection_name):
        client.collections.delete(config.smart_collection_name)
        logger.info("Dropped collection '%s'.", config.smart_collection_name)


def _smart_chunk_uuid(chunk: SmartTextChunk) -> str:
    """Deterministic UUID for a child chunk (source_file + child chunk_index)."""
    return str(
        uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.source_file}::smart::{chunk.chunk_index}")
    )


def store_smart_chunks(
    client: weaviate.WeaviateClient,
    chunks: List[SmartTextChunk],
    embeddings: List[List[float]],
    config: WeaviateConfig,
) -> int:
    """
    Batch-insert smart chunks with their context_window embedding vectors.

    Note: *embeddings* must be computed from chunk.context_window strings,
    not from chunk.text.  The pipeline is responsible for passing the right
    texts to the embedding model.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
        )

    collection = client.collections.get(config.smart_collection_name)
    stored = 0

    with collection.batch.fixed_size(batch_size=config.batch_size) as batch:
        for chunk, vector in zip(chunks, embeddings):
            batch.add_object(
                properties={
                    "company":           chunk.company,
                    "quarter":           chunk.quarter,
                    "year":              chunk.year,
                    "file_name":         chunk.file_name,
                    "source_file":       chunk.source_file,
                    "chunk_index":       chunk.chunk_index,
                    "parent_text":       chunk.parent_text,
                    "parent_id":         chunk.parent_id,
                    "section_title":     chunk.section_title,
                    "page_num":          chunk.page_num,
                    "token_count":       chunk.token_count,
                    "last_updated_date": chunk.last_updated_date,
                },
                vector=vector,
                uuid=_smart_chunk_uuid(chunk),
            )
            stored += 1

    logger.info(
        "Queued %d objects for insertion into '%s'.", stored, config.smart_collection_name
    )
    return stored


def get_total_count_smart(client: weaviate.WeaviateClient, config: WeaviateConfig) -> int:
    """Return total number of objects in the SecDocumentSmart collection."""
    collection = client.collections.get(config.smart_collection_name)
    result = collection.aggregate.over_all(total_count=True)
    return result.total_count


# ── DocumentChunk – semantic chunking schema, ingestion, verification ─────────

def ensure_schema_semantic(client: weaviate.WeaviateClient, config: WeaviateConfig) -> None:
    """
    Create the DocumentChunk collection for semantic chunking.

    Schema highlights vs SecDocumentSmart:
        text_for_search  – title-prefixed + overlap text; BM25-indexed AND vectorized
                           (embeddings computed externally with BGE-M3 and pushed)
        raw_content      – clean chunk text returned to the LLM; stored only
        content_hash     – SHA256 of raw_content; used as dedup key for UUIDs
        section_title    – stored for filtering but not BM25-indexed (already
                           embedded in text_for_search)
    """
    if client.collections.exists(config.semantic_collection_name):
        logger.info(
            "Collection '%s' already exists – skipping creation.",
            config.semantic_collection_name,
        )
        return

    client.collections.create(
        name=config.semantic_collection_name,
        description=(
            "SEC 10-Q semantic chunks. "
            "Vector = BGE-M3 embedding of text_for_search; raw_content returned to LLM."
        ),
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
        ),
        properties=[
            # Metadata fields – filterable but not BM25-searched
            Property(name="company",           data_type=DataType.TEXT, index_searchable=False),
            Property(name="quarter",           data_type=DataType.TEXT, index_searchable=False),
            Property(name="year",              data_type=DataType.INT),
            Property(name="file_name",         data_type=DataType.TEXT, index_searchable=False),
            Property(name="source_file",       data_type=DataType.TEXT, index_searchable=False),
            Property(name="chunk_index",       data_type=DataType.INT),
            Property(name="page_num",          data_type=DataType.INT),
            Property(name="token_count",       data_type=DataType.INT),
            Property(name="section_title",     data_type=DataType.TEXT, index_searchable=False),
            Property(name="content_hash",      data_type=DataType.TEXT, index_searchable=False),
            Property(name="last_updated_date", data_type=DataType.TEXT, index_searchable=False),
            # raw_content: LLM display only — not BM25, not vectorized
            Property(
                name="raw_content",
                data_type=DataType.TEXT,
                index_searchable=False,
            ),
            # text_for_search: title-prefixed + overlap — BM25-indexed for hybrid search
            Property(
                name="text_for_search",
                data_type=DataType.TEXT,
                index_searchable=True,
                tokenization=Tokenization.WORD,
            ),
        ],
    )
    logger.info("Created collection '%s'.", config.semantic_collection_name)


def drop_collection_semantic(client: weaviate.WeaviateClient, config: WeaviateConfig) -> None:
    """Drop the DocumentChunk collection (useful for clean re-ingest)."""
    if client.collections.exists(config.semantic_collection_name):
        client.collections.delete(config.semantic_collection_name)
        logger.info("Dropped collection '%s'.", config.semantic_collection_name)


def _semantic_chunk_uuid(chunk: SemanticTextChunk) -> str:
    """Deterministic UUID based on content_hash for content-level deduplication."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"semantic::{chunk.content_hash}"))


def store_semantic_chunks(
    client: weaviate.WeaviateClient,
    chunks: List[SemanticTextChunk],
    embeddings: List[List[float]],
    config: WeaviateConfig,
) -> int:
    """
    Batch-insert semantic chunks with their BGE-M3 embedding vectors.

    Note: *embeddings* must be computed from chunk.text_for_search strings
    (not raw_content) so that the stored vector matches the BM25-indexed field.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
        )

    collection = client.collections.get(config.semantic_collection_name)
    stored = 0

    with collection.batch.fixed_size(batch_size=config.batch_size) as batch:
        for chunk, vector in zip(chunks, embeddings):
            batch.add_object(
                properties={
                    "company":           chunk.company,
                    "quarter":           chunk.quarter,
                    "year":              chunk.year,
                    "file_name":         chunk.file_name,
                    "source_file":       chunk.source_file,
                    "chunk_index":       chunk.chunk_index,
                    "text_for_search":   chunk.text_for_search,
                    "raw_content":       chunk.raw_content,
                    "content_hash":      chunk.content_hash,
                    "section_title":     chunk.section_title,
                    "page_num":          chunk.page_num,
                    "token_count":       chunk.token_count,
                    "last_updated_date": chunk.last_updated_date,
                },
                vector=vector,
                uuid=_semantic_chunk_uuid(chunk),
            )
            stored += 1

    logger.info(
        "Queued %d objects for insertion into '%s'.", stored, config.semantic_collection_name
    )
    return stored


def get_total_count_semantic(client: weaviate.WeaviateClient, config: WeaviateConfig) -> int:
    """Return total number of objects in the DocumentChunk collection."""
    collection = client.collections.get(config.semantic_collection_name)
    result = collection.aggregate.over_all(total_count=True)
    return result.total_count
