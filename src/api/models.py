"""Pydantic request/response models for the RAG API."""
from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    # Optional metadata filters – narrow retrieval to specific filings
    company: Optional[str] = Field(default=None, description="Ticker symbol filter, e.g. 'AAPL'")
    year: Optional[int] = Field(default=None, description="Filing year filter, e.g. 2023")
    quarter: Optional[str] = Field(default=None, description="Quarter filter, e.g. 'Q2'")
    # Retrieval engine and chunking strategy
    engine: str = Field(
        default="custom",
        description="Retrieval engine: 'custom' (direct Weaviate client) or 'llamaindex'",
    )
    chunking_strategy: Optional[str] = Field(
        default=None,
        description=(
            "Which ingested collection to query: "
            "'basic' → SecDocument, "
            "'parent_child' → SecDocumentSmart, "
            "'semantic' → DocumentChunk (BGE-M3). "
            "When null, falls back to use_smart for compatibility."
        ),
    )
    use_smart: bool = Field(
        default=False,
        description=(
            "Legacy flag. True maps to chunking_strategy='parent_child'. "
            "Ignored when chunking_strategy is provided."
        ),
    )
    # Optional re-ranking after initial retrieval (None = disabled)
    rerank_mode: Optional[str] = Field(
        default=None,
        description=(
            "Re-rank over-fetched candidates before generation. "
            "'llm': one gpt-4o-mini call ranks all candidates. "
            "'cross_encoder': local sentence-transformers (no API cost). "
            "null/omit to skip re-ranking."
        ),
    )
    # Filter extraction mode when company/year/quarter are not passed explicitly
    filter_mode: str = Field(
        default="llm",
        description=(
            "'llm' (default): extract company/year/quarter from query text via LLM. "
            "'regex': rule-based extraction, no LLM cost. "
            "Ignored if company/year/quarter are provided explicitly."
        ),
    )
    # Per-request retrieval mode override (falls back to RETRIEVAL_MODE env var)
    retrieval_mode: Optional[str] = Field(
        default=None,
        description="Override retrieval mode: 'semantic' or 'hybrid'. Defaults to server config.",
    )
    retrieval_alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="BM25/vector balance for hybrid mode (0=BM25, 1=vector). Defaults to server config.",
    )


class SourceDocument(BaseModel):
    company: str
    quarter: str
    year: int
    source_file: str
    chunk_index: int
    page_start: int
    page_end: int
    score: float = Field(description="Similarity score in [0, 1], higher = more relevant")
    text_preview: str = Field(description="First 200 characters of the chunk")


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    query: str
    chunks_retrieved: int
