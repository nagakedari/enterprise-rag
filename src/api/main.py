"""
Enterprise RAG – FastAPI application.

Endpoints:
    POST /chat   – answer a question with an agentic tool-calling loop
                   (vector search + Python calculators, model-driven)
    GET  /health – liveness check

Run locally:
    uvicorn src.api.main:app --reload --port 8000
"""
import functools
import logging
import time
from datetime import date
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Must run before any local (src.*) import — several modules instantiate a
# Config() at import time (e.g. src/api/evaluation_service.py), which reads
# env vars immediately, so .env has to be loaded first.
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.evaluation_routes import router as evaluation_router
from src.agentic import run_agent
from src.api.models import ChatRequest, ChatResponse, SourceDocument, ToolCallTrace
from src.config import Config
from src.retrieval.retriever import FilingPeriod, RetrievedChunk, list_filing_periods, retrieve
from src.retrieval.llamaindex_retriever import retrieve_llamaindex
from src.retrieval.reranker import rerank

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

# Vite dev server origins — local dev only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(evaluation_router)

# Instantiate config once at startup (reads env vars)
_config = Config()


@app.get("/health")
def health():
    return {"status": "ok"}


class RetrievalError(Exception):
    """Vector-store retrieval failed inside the agent loop (→ HTTP 502)."""


def _with_server_defaults(request: ChatRequest) -> ChatRequest:
    """
    Fill every retrieval setting the caller left unset from _config.chat
    (best evaluation config: parent_child / hybrid / alpha 0.25 / top_k 10 /
    cross-encoder re-rank to 6), so _retrieve_for_request only ever sees
    concrete settings. rerank_mode="none" turns re-ranking off.
    """
    defaults = _config.chat
    chunking_strategy = request.chunking_strategy or (
        "parent_child" if request.use_smart else defaults.chunking_strategy
    )
    rerank_mode = request.rerank_mode if request.rerank_mode is not None else defaults.rerank_mode
    if rerank_mode == "none":
        rerank_mode = None
    return request.model_copy(update={
        "chunking_strategy": chunking_strategy,
        "retrieval_mode": request.retrieval_mode or defaults.retrieval_mode,
        "top_k": request.top_k or defaults.top_k,
        "retrieval_alpha": (
            request.retrieval_alpha if request.retrieval_alpha is not None
            else defaults.retrieval_alpha
        ),
        "rerank_mode": rerank_mode,
        "rerank_top_k": request.rerank_top_k or defaults.rerank_top_k,
    })


def _collection_for(chunking_strategy: str) -> str:
    if chunking_strategy == "semantic":
        return _config.weaviate.semantic_collection_name
    if chunking_strategy == "parent_child":
        return _config.weaviate.smart_collection_name
    return _config.weaviate.collection_name


# Filing catalog for the agent's planner, read live from Weaviate and cached
# briefly so newly ingested filings appear without a restart.
_CATALOG_TTL_SECONDS = 300
_catalog_cache: Dict[str, Tuple[float, List[FilingPeriod]]] = {}


def _filing_catalog(request: ChatRequest) -> List[FilingPeriod]:
    # LlamaIndex collections don't carry source_file/DATE properties; the
    # custom collection for the same strategy describes the same filings.
    collection_name = _collection_for(request.chunking_strategy)
    now = time.monotonic()
    cached = _catalog_cache.get(collection_name)
    if cached and now - cached[0] < _CATALOG_TTL_SECONDS:
        rows = cached[1]
    else:
        try:
            rows = list_filing_periods(_config, collection_name)
            _catalog_cache[collection_name] = (now, rows)
        except Exception:
            # The agent still works without it — it just has to guess periods.
            logger.exception("Could not load filing catalog from %s", collection_name)
            rows = cached[1] if cached else []
    if request.company:
        rows = [r for r in rows if r.company == request.company]
    return rows


def _retrieve_for_request(
    request: ChatRequest,
    query: str,
    company: Optional[str] = None,
    year: Optional[int] = None,
    quarter: Optional[str] = None,
    period_date: Optional[date] = None,
) -> List[RetrievedChunk]:
    """
    One retrieval pass using the request's retrieval settings (engine,
    chunking strategy, mode/alpha, top_k, rerank) — `request` must already
    be resolved by _with_server_defaults. This is the `search_fn` the agent
    calls — possibly several times per request, each with its own query and
    filters.

    Filters supplied explicitly on the request are pinned: they override
    whatever the agent asks for, so a caller-scoped request can't drift
    outside its scope.
    """
    company = request.company or company
    year = request.year or year
    quarter = request.quarter or quarter

    logger.info(
        "Agent search: query=%r filters=(company=%s year=%s quarter=%s period_date=%s)",
        query, company, year, quarter, period_date,
    )
    chunking_strategy = request.chunking_strategy
    fetch_k = request.top_k * 3 if request.rerank_mode else request.top_k
    final_k = request.rerank_top_k if request.rerank_mode else request.top_k
    try:
        if request.engine == "llamaindex":
            chunks = retrieve_llamaindex(
                query=query,
                config=_config,
                top_k=fetch_k,
                company=company,
                year=year,
                quarter=quarter,
                use_smart=(chunking_strategy == "parent_child"),
                mode=request.retrieval_mode,
                alpha=request.retrieval_alpha,
                period_date=period_date,
            )
        else:
            chunks = retrieve(
                query=query,
                config=_config,
                top_k=fetch_k,
                company=company,
                year=year,
                quarter=quarter,
                collection_name=_collection_for(chunking_strategy),
                mode=request.retrieval_mode,
                alpha=request.retrieval_alpha,
                period_date=period_date,
            )

        if request.rerank_mode and len(chunks) > final_k:
            chunks = rerank(
                question=query,
                chunks=chunks,
                top_k=final_k,
                mode=request.rerank_mode,
                api_key=_config.generation.api_key,
                model=_config.generation.model,
            )
    except Exception as exc:
        raise RetrievalError(str(exc)) from exc
    return chunks


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Answer a question agentically (src/agentic/agent.py): the model decides
    which searches to run (query + company/year/quarter filters), which
    FinancialAnalyticsTools calculator to call with the figures it found,
    and when it has enough to write the final cited answer. Each search runs
    through this request's retrieval settings via _retrieve_for_request.
    Retrieval settings the caller leaves unset come from _config.chat.
    """
    request = _with_server_defaults(request)
    logger.info(
        "Chat request: query=%r engine=%s chunking_strategy=%s retrieval_mode=%s "
        "alpha=%s top_k=%d rerank=%s(top %d) pinned_filters=(company=%s year=%s quarter=%s)",
        request.query, request.engine, request.chunking_strategy,
        request.retrieval_mode or _config.retrieval.mode, request.retrieval_alpha,
        request.top_k, request.rerank_mode, request.rerank_top_k,
        request.company, request.year, request.quarter,
    )

    try:
        result = run_agent(
            query=request.query,
            search_fn=functools.partial(_retrieve_for_request, request),
            config=_config.agent,
            filings=_filing_catalog(request),
        )
    except RetrievalError as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=502, detail=f"Retrieval error: {exc}") from exc
    except Exception as exc:
        logger.exception("Agent failed")
        raise HTTPException(status_code=502, detail=f"Agent error: {exc}") from exc

    chunks = result.sources
    answer = result.answer

    # ── Build response ─────────────────────────────────────────────────────────
    def source_document(c: RetrievedChunk, repeats: Optional[List[int]] = None) -> SourceDocument:
        return SourceDocument(
            company=c.company,
            quarter=c.quarter,
            year=c.year,
            source_file=c.source_file,
            chunk_index=c.chunk_index,
            page_start=c.page_start,
            page_end=c.page_end,
            score=round(c.score, 4),
            text_preview=c.text[:200],
            repeats_sources=repeats,
        )

    sources = [source_document(c) for c in chunks]
    repeated_sources = [source_document(c, repeats) for c, repeats in result.repeated_sources]

    return ChatResponse(
        answer=answer,
        sources=sources,
        query=request.query,
        chunks_retrieved=len(chunks),
        repeated_sources=repeated_sources,
        tool_calls=[
            ToolCallTrace(name=r.name, arguments=r.arguments, result=r.result, error=r.error)
            for r in result.tool_calls
        ],
        agent_iterations=result.iterations,
        agent_plan=result.plan,
        agent_topics=result.topics,
    )


if __name__ == "__main__":
    # import sys
    # from pathlib import Path

    # sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
