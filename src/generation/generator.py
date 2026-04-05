"""
Answer generation via OpenAI Chat Completions.

Takes a user query and a list of retrieved chunks, builds a RAG prompt,
and returns the model's answer.

Phase-1: single-turn generation, no conversation history, no streaming.
"""
import logging
from typing import List

import openai

from src.config import GenerationConfig
from src.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a financial analyst assistant that answers questions about \
SEC 10-Q filings.

Answer the user's question using ONLY the context provided below. \
Be concise and precise. If the context does not contain enough information \
to answer the question, say "I don't have enough information in the \
retrieved documents to answer that."

Do not fabricate numbers, dates, or facts not present in the context."""


def _build_context_block(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[Source {i}: {chunk.company} {chunk.quarter} {chunk.year} "
            f"| {chunk.source_file} | pages {chunk.page_start}–{chunk.page_end}]"
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

    client = openai.OpenAI(api_key=config.api_key)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    answer = response.choices[0].message.content or ""
    logger.info("Generated answer (%d chars)", len(answer))
    return answer.strip()
