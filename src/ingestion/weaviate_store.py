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
from weaviate.classes.config import Configure, DataType, Property, VectorDistances

from src.config import WeaviateConfig
from src.ingestion.chunker import TextChunk

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
            Property(name="company",     data_type=DataType.TEXT),
            Property(name="quarter",     data_type=DataType.TEXT),
            Property(name="year",        data_type=DataType.INT),
            Property(name="source_file", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="chunk_text",  data_type=DataType.TEXT),
            Property(name="page_start",  data_type=DataType.INT),
            Property(name="page_end",    data_type=DataType.INT),
            Property(name="token_count", data_type=DataType.INT),
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
