"""
Re-rank retrieved chunks by relevance to the question.

Why re-ranking
--------------
Initial retrieval (vector / hybrid) optimises for approximate nearest-neighbour
similarity, which has two weaknesses for financial Q&A:

1. Boilerplate inflation – SEC filings repeat legal disclaimers and table
   headers across many chunks.  These score well on both BM25 (shared keywords)
   and vector similarity (similar embedding) even though they don't answer the
   question.

2. Topic dilution in parent chunks – a 1000-token parent chunk may be relevant
   to the question on one sentence but contain three unrelated financial topics.
   Similarity search can't distinguish signal from noise at the sub-chunk level.

Re-ranking solves this by scoring each candidate chunk with a model that sees
BOTH the question and the chunk text at the same time (cross-attention), instead
of comparing pre-computed embeddings independently.

Two strategies
--------------
"llm"          One LLM call that receives all candidates and returns a ranked
               ordering.  Uses a listwise prompt so the model can compare
               chunks against each other (not just against the question).
               More accurate, especially for financial domain language.
               Cost: ~1 gpt-4o-mini call per retrieval.

"cross_encoder" Local sentence-transformers cross-encoder (no API cost after
               install).  Requires: pip install sentence-transformers
               Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
               Fast on CPU (~50 ms for 15 candidates).
               Trade-off: general-purpose model, not domain-tuned for SEC.

Both strategies:
  - Accept a list of RetrievedChunk objects
  - Return the top_k most relevant ones, reordered by re-rank score
  - Fall back gracefully (return original order) on any error

Usage
-----
    from src.retrieval.reranker import rerank

    # Step 1: over-fetch candidates
    candidates = retrieve(query, config, top_k=top_k * 3, ...)

    # Step 2: re-rank, keep best top_k
    chunks = rerank(
        question=query,
        chunks=candidates,
        top_k=top_k,
        mode="llm",               # or "cross_encoder"
        api_key=config.generation.api_key,
        model=config.generation.model,
    )
"""
import logging
from typing import List, Literal, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM listwise re-ranker
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """\
You are a passage relevance judge for a financial question-answering system \
over SEC 10-Q filings.

You will be given a QUESTION and a numbered list of PASSAGES retrieved from \
SEC documents.

Your task: return a JSON array of passage numbers ordered from MOST relevant \
to LEAST relevant for answering the question.

Relevance criteria (in priority order):
1. The passage directly contains facts, figures, or explanations that answer \
the question.
2. The passage is from the correct company, fiscal year, and quarter.
3. The passage provides supporting context that helps interpret the answer.
4. Boilerplate text (legal disclaimers, forward-looking statement warnings, \
table headers) is NOT relevant even if it mentions the company name.

Rules:
- Include ALL passage numbers in your output (no omissions).
- Return ONLY a JSON array of integers, e.g.: [3, 1, 5, 2, 4]
- No explanation, no markdown, no extra text.
"""


def _rerank_llm(
    question: str,
    chunks: list,
    top_k: int,
    api_key: str,
    model: str,
) -> list:
    import json
    import openai

    if not chunks:
        return chunks

    # Build numbered passage list (1-indexed for readability in the prompt)
    passages = "\n\n".join(
        f"[{i + 1}] {c.text[:600]}"   # cap each passage to avoid token overflow
        for i, c in enumerate(chunks)
    )
    user_msg = f"QUESTION: {question}\n\nPASSAGES:\n{passages}"

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "").strip()

        # Model may return {"ranking": [3,1,...]} or just [3,1,...] as a JSON object
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            ranking = parsed
        elif isinstance(parsed, dict):
            # Find the first list value in the dict
            ranking = next(
                (v for v in parsed.values() if isinstance(v, list)), None
            )
            if ranking is None:
                raise ValueError(f"No list found in LLM response: {parsed}")
        else:
            raise ValueError(f"Unexpected LLM response shape: {type(parsed)}")

        # Convert 1-indexed ranks → 0-indexed, filter out-of-range values
        n = len(chunks)
        indices = [int(r) - 1 for r in ranking if 1 <= int(r) <= n]

        # Append any chunk the model forgot to include (preserve all chunks)
        seen = set(indices)
        indices += [i for i in range(n) if i not in seen]

        reranked = [chunks[i] for i in indices[:top_k]]
        logger.debug(
            "LLM re-ranker: %d candidates → top %d  order=%s",
            n, top_k, [i + 1 for i in indices[:top_k]],
        )
        return reranked

    except Exception as exc:
        logger.warning("LLM re-ranker failed (%s) — returning original order.", exc)
        return chunks[:top_k]


# ---------------------------------------------------------------------------
# Cross-encoder re-ranker
# ---------------------------------------------------------------------------

_DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _rerank_cross_encoder(
    question: str,
    chunks: list,
    top_k: int,
    model_name: str,
) -> list:
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except ImportError:
        logger.warning(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers\n"
            "Falling back to original retrieval order."
        )
        return chunks[:top_k]

    try:
        encoder = CrossEncoder(model_name)
        pairs = [(question, c.text[:512]) for c in chunks]
        scores = encoder.predict(pairs)

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )
        reranked = [c for _, c in ranked[:top_k]]
        logger.debug(
            "Cross-encoder re-ranker: %d candidates → top %d  top_score=%.4f",
            len(chunks), top_k, ranked[0][0] if ranked else 0,
        )
        return reranked

    except Exception as exc:
        logger.warning("Cross-encoder re-ranker failed (%s) — returning original order.", exc)
        return chunks[:top_k]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(
    question: str,
    chunks: list,
    top_k: int,
    mode: Literal["llm", "cross_encoder"] = "llm",
    api_key: str = "",
    model: str = "gpt-4o-mini",
    cross_encoder_model: str = _DEFAULT_CROSS_ENCODER_MODEL,
) -> list:
    """
    Re-rank *chunks* by relevance to *question* and return the best *top_k*.

    Args:
        question:             User's question.
        chunks:               Candidate chunks from an initial retrieval call
                              (typically top_k * 3 over-fetched candidates).
        top_k:                Number of chunks to return after re-ranking.
        mode:                 ``"llm"``           — listwise LLM re-ranking
                                                   (one API call, domain-aware).
                              ``"cross_encoder"`` — local sentence-transformers
                                                   model (no API cost).
        api_key:              OpenAI key. Required for ``mode="llm"``.
        model:                LLM model for ``mode="llm"`` (default: gpt-4o-mini).
        cross_encoder_model:  HuggingFace model ID for ``mode="cross_encoder"``.

    Returns:
        List of at most *top_k* RetrievedChunk objects, best-first.
        Falls back to original retrieval order on any error so retrieval
        always succeeds even if re-ranking fails.
    """
    if not chunks:
        return chunks

    if len(chunks) <= top_k:
        # Nothing to re-rank — already at or below target count
        return chunks

    logger.info(
        "Re-ranking %d candidates → top %d  mode=%s",
        len(chunks), top_k, mode,
    )

    if mode == "cross_encoder":
        return _rerank_cross_encoder(question, chunks, top_k, cross_encoder_model)

    # Default: LLM
    if not api_key:
        logger.warning(
            "rerank called with mode='llm' but no api_key — "
            "falling back to original order."
        )
        return chunks[:top_k]

    return _rerank_llm(question, chunks, top_k, api_key, model)
