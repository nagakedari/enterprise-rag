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
from src.retrieval.llamaindex_retriever import retrieve_llamaindex
from src.retrieval.query_filters import extract_query_filters
from src.retrieval.reranker import rerank

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
    # If the caller didn't supply explicit filters, extract them from the query.
    company  = request.company
    year     = request.year
    quarter  = request.quarter
    if company is None and year is None and quarter is None:
        filters = extract_query_filters(
            question=request.query,
            mode=request.filter_mode,
            api_key=_config.generation.api_key,
            model=_config.generation.model,
        )
        company  = filters["company"]
        year     = filters["year"]
        quarter  = filters["quarter"]

    logger.info(
        "engine=%s  use_smart=%s  retrieval_mode=%s  alpha=%s  "
        "filters=(company=%s year=%s quarter=%s)  filter_mode=%s",
        request.engine, request.use_smart, request.retrieval_mode, request.retrieval_alpha,
        company, year, quarter, request.filter_mode,
    )
    fetch_k = request.top_k * 3 if request.rerank_mode else request.top_k
    try:
        # Resolve effective chunking strategy (explicit wins over legacy use_smart)
        chunking_strategy = request.chunking_strategy or (
            "parent_child" if request.use_smart else "basic"
        )

        if request.engine == "llamaindex":
            chunks = retrieve_llamaindex(
                query=request.query,
                config=_config,
                top_k=fetch_k,
                company=company,
                year=year,
                quarter=quarter,
                use_smart=(chunking_strategy == "parent_child"),
                mode=request.retrieval_mode,
                alpha=request.retrieval_alpha,
            )
        else:
            if chunking_strategy == "semantic":
                collection_name = _config.weaviate.semantic_collection_name
            elif chunking_strategy == "parent_child":
                collection_name = _config.weaviate.smart_collection_name
            else:
                collection_name = _config.weaviate.collection_name
            chunks = retrieve(
                query=request.query,
                config=_config,
                top_k=fetch_k,
                company=company,
                year=year,
                quarter=quarter,
                collection_name=collection_name,
                mode=request.retrieval_mode,
                alpha=request.retrieval_alpha,
            )

        if request.rerank_mode and len(chunks) > request.top_k:
            chunks = rerank(
                question=request.query,
                chunks=chunks,
                top_k=request.top_k,
                mode=request.rerank_mode,
                api_key=_config.generation.api_key,
                model=_config.generation.model,
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
