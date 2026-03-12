"""
Central configuration for the ingestion pipeline.
All values are read from environment variables with sensible defaults.
Load your .env file before importing this module, e.g.:
    from dotenv import load_dotenv; load_dotenv()
"""
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ChunkConfig:
    """Token-based chunking parameters."""
    min_tokens: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_MIN_TOKENS", "500"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_MAX_TOKENS", "800"))
    )
    overlap_tokens: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))
    )
    # cl100k_base = tokenizer used by text-embedding-3-* and gpt-4
    encoding: str = "cl100k_base"


@dataclass
class WeaviateConfig:
    """Connection settings for Weaviate."""
    http_host: str = field(
        default_factory=lambda: os.getenv("WEAVIATE_HOST", "localhost")
    )
    http_port: int = field(
        default_factory=lambda: int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
    )
    grpc_port: int = field(
        default_factory=lambda: int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    )
    collection_name: str = "SecDocument"
    batch_size: int = 100


@dataclass
class EmbeddingConfig:
    """OpenAI embedding settings."""
    model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    # How many texts to send per API call
    api_batch_size: int = 100
    # Dimension of text-embedding-3-small
    dimensions: int = 1536


@dataclass
class Config:
    """Top-level config aggregating all sub-configs."""
    docs_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "DOCS_PATH",
                "/Users/manjusri/learning/generative_ai/KG-RAG-datasets/sec-10-q/data/v1/docs",
            )
        )
    )
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    weaviate: WeaviateConfig = field(default_factory=WeaviateConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
