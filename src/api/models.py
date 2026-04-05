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
