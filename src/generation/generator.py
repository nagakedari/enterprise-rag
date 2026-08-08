"""
Answer generation via OpenAI Chat Completions.

Takes a user query and a list of retrieved chunks, builds a RAG prompt,
and returns the model's answer.

Phase-1: single-turn generation, no conversation history, no streaming.

Prompt-caching notes
--------------------
OpenAI automatically caches prompt prefixes that are ≥ 1,024 tokens and
identical across requests, charging 50 % of normal input-token cost for
cache hits (gpt-4o / gpt-4o-mini as of 2024-05).

Two design decisions here exploit that:

  1. _SYSTEM_PROMPT is a module-level constant — it is byte-for-byte
     identical on every call, so OpenAI can cache it as the leading prefix.

  2. The OpenAI client is created once via _get_client() (lru_cache) instead
     of inside generate() — avoids a new TCP connection per call and lets the
     SDK reuse the same HTTP session, which is a prerequisite for the server
     recognising the prefix as a cache hit.

The generated answer's log line now reports `cached=N` tokens so you can
verify cache savings in the logs.
"""
import functools
import logging
from typing import List

import openai

from src.config import GenerationConfig
from src.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


# ── Cached client factory ─────────────────────────────────────────────────────
# CHANGED: moved client creation out of generate() so the same OpenAI client
# (and its underlying HTTP connection pool) is reused across all calls.
# lru_cache keyed on api_key means one client per key, created lazily on first use.
@functools.lru_cache(maxsize=4)
def _get_client(api_key: str) -> openai.OpenAI:
    logger.debug("Creating new OpenAI client (cached for subsequent calls).")
    return openai.OpenAI(api_key=api_key)

_SYSTEM_PROMPT = """You are a financial analyst assistant that answers questions about \
SEC 10-Q filings.

STRICT RULES — follow every rule for every answer:

1. READ ALL SOURCES FIRST.
   Before writing a single word of your answer, read every numbered source block \
in the context. Do not stop at the first relevant source. Facts relevant to the \
question may be spread across multiple chunks.

2. CITE EVERY CLAIM.
   After each factual claim, add an inline citation: (Company, Quarter Year, p.PAGE).
   Example: "Total net sales were $82,959M (AAPL, Q3 2022, p.4)."
   If a claim is supported by more than one source, list all: \
(AAPL Q3 2022 p.4; AAPL Q1 2023 p.6).

3. HANDLE MISSING INFORMATION EXPLICITLY.
   Use ONLY facts present in the provided sources. If the context does not contain \
enough information to fully answer the question, state exactly what is missing:
   "The context does not include [specific fact]. Based on available sources: …"
   Never infer, extrapolate, or fill gaps from general knowledge.

4. COMPARISON QUESTIONS — two-step approach.
   When the question asks you to compare quarters, years, or companies:
   Step A — Extract: list the relevant figure for each entity from its source.
   Step B — Compare: produce the comparison only after all figures are listed.
   This prevents anchoring on the first entity and ignoring the rest."""


def _build_context_block(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[Source {i}: {chunk.company} {chunk.quarter} {chunk.year} "
            f"| {chunk.source_file} | p.{chunk.page_start}–{chunk.page_end}]"
        )
        parts.append(f"{header}\n{chunk.text.strip()}")
    return "\n\n---\n\n".join(parts)


def generate(
    query: str,
    chunks: List[RetrievedChunk],
    config: GenerationConfig,
) -> str:
    """
    Call the LLM with the retrieved context and return its answer.

    Args:
        query:   The user's original question.
        chunks:  Retrieved context chunks (ordered by similarity).
        config:  Generation model settings.

    Returns:
        The model's answer as a plain string.
    """
    if not chunks:
        return "No relevant documents were found to answer your question."

    context_block = _build_context_block(chunks)
    user_message = f"Context:\n\n{context_block}\n\nQuestion: {query}"

    logger.info(
        "Calling %s with %d context chunks (total ~%d tokens)",
        config.model,
        len(chunks),
        sum(c.token_count for c in chunks),
    )

    # CHANGED: reuse cached client instead of creating a new one per call.
    client = _get_client(config.api_key)

    # The system message carries _SYSTEM_PROMPT — a module-level constant that
    # is byte-for-byte identical on every request.  OpenAI's automatic prompt
    # caching treats the leading prefix as a cache key, so keeping this message
    # static (no per-query interpolation) maximises cache hit probability.
    # Cache hits are billed at 50 % of normal input-token cost.
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},  # static — cacheable prefix
            {"role": "user",   "content": user_message},    # dynamic — context + question
        ],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    answer = response.choices[0].message.content or ""

    # CHANGED: log cached_tokens from the usage object so cache savings are visible.
    # cached_tokens > 0 means OpenAI served that portion from its KV cache at half price.
    usage = response.usage
    cached = (
        getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    )
    logger.info(
        "Generated answer (%d chars) | tokens: prompt=%d cached=%d (%.0f%%) completion=%d",
        len(answer),
        usage.prompt_tokens,
        cached,
        (cached / usage.prompt_tokens * 100) if usage.prompt_tokens else 0,
        usage.completion_tokens,
    )
    return answer.strip()
