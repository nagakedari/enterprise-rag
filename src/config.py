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
class SmartChunkConfig:
    """
    Parent-child chunking parameters (Phase 2).

    Parent chunks (~1000 tokens) are returned to the LLM for generation.
    Child chunks (~300 tokens) are embedded for precise vector retrieval.
    Each child stores a reference to its parent so the retriever can return
    the richer parent context after matching on the smaller child vector.
    """
    # Parent chunk: larger unit of text returned to the LLM
    parent_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("SMART_PARENT_MAX_TOKENS", "1000"))
    )
    parent_overlap_tokens: int = field(
        default_factory=lambda: int(os.getenv("SMART_PARENT_OVERLAP_TOKENS", "100"))
    )
    # Child chunk: smaller unit embedded for retrieval
    child_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("SMART_CHILD_MAX_TOKENS", "300"))
    )
    child_overlap_tokens: int = field(
        default_factory=lambda: int(os.getenv("SMART_CHILD_OVERLAP_TOKENS", "50"))
    )
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
    smart_collection_name: str = field(
        default_factory=lambda: os.getenv("WEAVIATE_SMART_COLLECTION", "SecDocumentSmart")
    )
    # LlamaIndex-managed collections (separate from custom pipeline collections)
    llamaindex_collection_name: str = field(
        default_factory=lambda: os.getenv("WEAVIATE_LI_COLLECTION", "SecDocumentLI")
    )
    llamaindex_smart_collection_name: str = field(
        default_factory=lambda: os.getenv("WEAVIATE_LI_SMART_COLLECTION", "SecDocumentSmartLI")
    )
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
class GenerationConfig:
    """OpenAI Chat Completions settings for answer generation."""
    model: str = field(
        default_factory=lambda: os.getenv("GENERATION_MODEL", "gpt-4o-mini")
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("GENERATION_MAX_TOKENS", "1024"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("GENERATION_TEMPERATURE", "0.0"))
    )


@dataclass
class EvaluationConfig:
    """
    LLM settings for the evaluation / judge step.

    Intentionally separate from GenerationConfig so that evaluation uses a
    stronger model than the one being tested — avoids the 'LLM judging itself'
    self-serving bias.

    Default judge: gpt-4o  (stronger than gpt-4o-mini used for generation)
    Override via EVALUATION_MODEL env var.
    """
    model: str = field(
        default_factory=lambda: os.getenv("EVALUATION_MODEL", "gpt-4o")
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    temperature: float = 0.0


@dataclass
class RetrievalConfig:
    """Vector retrieval settings."""
    top_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "5"))
    )
    # "semantic" → pure cosine similarity (near_vector)
    # "hybrid"   → BM25 + cosine via Weaviate hybrid search
    mode: str = field(
        default_factory=lambda: os.getenv("RETRIEVAL_MODE", "hybrid")
    )
    # alpha controls BM25 / vector balance in hybrid mode:
    #   0.0 = pure BM25 (lexical only)
    #   0.5 = equal weight  (default)
    #   1.0 = pure vector   (same as semantic)
    alpha: float = field(
        default_factory=lambda: float(os.getenv("RETRIEVAL_HYBRID_ALPHA", "0.5"))
    )


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
    smart_chunk: SmartChunkConfig = field(default_factory=SmartChunkConfig)
    weaviate: WeaviateConfig = field(default_factory=WeaviateConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    # Persistent docstore for LlamaIndex AutoMergingRetriever (smart mode)
    llamaindex_docstore_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("LLAMAINDEX_DOCSTORE_PATH", "./storage/llamaindex_docstore")
        )
    )
