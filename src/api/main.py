"""
Enterprise RAG – FastAPI application.

Endpoints:
    POST /chat   – answer a question using vector retrieval + LLM generation
    GET  /health – liveness check

Run locally:
    uvicorn src.api.main:app --reload --port 8000
"""
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from src.api.models import ChatRequest, ChatResponse, SourceDocument
from src.config import Config
from src.generation.generator import generate
from src.retrieval.retriever import retrieve

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise RAG API",
    description="Answer questions over SEC 10-Q filings via vector retrieval + LLM generation.",
    version="1.0.0",
)

# Instantiate config once at startup (reads env vars)
_config = Config()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Answer a question by:
    1. Embedding the query and retrieving the top-k most similar chunks from Weaviate.
    2. Passing the retrieved context + query to an LLM to generate an answer.
    """
    logger.info("Chat request: query=%r  top_k=%d  filters=(company=%s, year=%s, quarter=%s)",
                request.query, request.top_k, request.company, request.year, request.quarter)

    # ── Retrieval ──────────────────────────────────────────────────────────────
    try:
        chunks = retrieve(
            query=request.query,
            config=_config,
            top_k=request.top_k,
            company=request.company,
            year=request.year,
            quarter=request.quarter,
        )
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=502, detail=f"Retrieval error: {exc}") from exc

    # ── Generation ─────────────────────────────────────────────────────────────
    try:
        answer = generate(
            query=request.query,
            chunks=chunks,
            config=_config.generation,
        )
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(status_code=502, detail=f"Generation error: {exc}") from exc

    # ── Build response ─────────────────────────────────────────────────────────
    sources = [
        SourceDocument(
            company=c.company,
            quarter=c.quarter,
            year=c.year,
            source_file=c.source_file,
            chunk_index=c.chunk_index,
            page_start=c.page_start,
            page_end=c.page_end,
            score=round(c.score, 4),
            text_preview=c.text[:200],
        )
        for c in chunks
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        query=request.query,
        chunks_retrieved=len(chunks),
    )


if __name__ == "__main__":
    # import sys
    # from pathlib import Path

    # sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
