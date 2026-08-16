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

Diversity selection (diversity_mode)
-------------------------------------
After scoring all candidates the default behaviour is "sort descending → take
top_k".  This is optimal for individual chunk relevance but hurts context_recall
on Multi-Doc questions because the cross-encoder may pick 4 chunks from one
quarter and only 2 from the other.

Two alternatives replace the final selection step:

"mmr"            Maximal Marginal Relevance.  Iteratively selects chunks that
                 are relevant to the query *and* dissimilar to already-chosen
                 chunks.  Uses the reranker score as the relevance signal and
                 TF-IDF cosine similarity between chunk texts as the diversity
                 signal.  mmr_lambda controls the relevance/diversity trade-off
                 (1.0 = pure relevance, 0.0 = pure diversity; default 0.5).
                 Requires: scikit-learn

"metadata_slots" Guarantees proportional slot coverage per (company, quarter,
                 year) entity present in the candidate pool.  Each unique entity
                 gets floor(top_k / n_entities) guaranteed slots filled by its
                 highest-scoring chunks; remaining slots go to the overall
                 highest-scoring unchosen chunks.  No extra dependencies.
                 Best when Multi-Doc questions need facts from two or more
                 filings that the reranker would otherwise crowd out.

Usage
-----
    from src.retrieval.reranker import rerank

    # Step 1: over-fetch candidates
    candidates = retrieve(query, config, top_k=top_k * 3, ...)

    # Step 2: re-rank with MMR diversity
    chunks = rerank(
        question=query,
        chunks=candidates,
        top_k=top_k,
        mode="cross_encoder",
        diversity_mode="mmr",
        mmr_lambda=0.5,
    )

    # Step 3: or use metadata slot allocation
    chunks = rerank(
        question=query,
        chunks=candidates,
        top_k=top_k,
        mode="cross_encoder",
        diversity_mode="metadata_slots",
    )
"""
import logging
import math
from typing import Any, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diversity selection helpers
# ---------------------------------------------------------------------------

def _select_mmr(
    scored_chunks: List[Tuple[float, Any]],
    top_k: int,
    lambda_: float = 0.5,
) -> List[Any]:
    """
    Maximal Marginal Relevance selection.

    Iteratively picks the chunk that maximises:
        lambda_ * norm_relevance(chunk)
        - (1 - lambda_) * max_similarity(chunk, already_selected)

    Args:
        scored_chunks: (relevance_score, chunk) pairs, sorted best-first.
                       The relevance score comes from the reranker (CE score
                       or LLM rank pseudo-score).
        top_k:         Number of chunks to return.
        lambda_:       1.0 = pure relevance (degenerates to top-k sort).
                       0.0 = pure diversity.  Default 0.5.

    Returns:
        top_k chunks in MMR selection order (best first).
    """
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        logger.warning(
            "MMR requires scikit-learn: pip install scikit-learn. "
            "Falling back to top-k sort."
        )
        return [c for _, c in scored_chunks[:top_k]]

    n = len(scored_chunks)
    top_k = min(top_k, n)
    if top_k == 0:
        return []

    # Normalise relevance scores to [0, 1] so they're comparable across runs.
    raw = np.array([s for s, _ in scored_chunks], dtype=float)
    lo, hi = raw.min(), raw.max()
    norm_scores = (raw - lo) / (hi - lo) if hi > lo else np.ones(n)

    # TF-IDF cosine similarity matrix — captures lexical redundancy between
    # chunks (e.g. repeated income-statement table rows have high overlap).
    texts = [c.text[:1000] for _, c in scored_chunks]
    try:
        vec = TfidfVectorizer(max_features=2000, stop_words="english")
        tfidf = vec.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf)
    except Exception as exc:
        logger.warning("MMR: TF-IDF failed (%s) — falling back to top-k sort.", exc)
        return [c for _, c in scored_chunks[:top_k]]

    selected: List[int] = []
    remaining = list(range(n))

    while len(selected) < top_k and remaining:
        if not selected:
            best = max(remaining, key=lambda i: norm_scores[i])
        else:
            best = max(
                remaining,
                key=lambda i: (
                    lambda_ * norm_scores[i]
                    - (1 - lambda_) * float(np.max(sim_matrix[i, selected]))
                ),
            )
        selected.append(best)
        remaining.remove(best)

    logger.debug(
        "MMR selected indices (lambda=%.2f): %s",
        lambda_,
        [(i, round(float(norm_scores[i]), 3)) for i in selected],
    )
    return [scored_chunks[i][1] for i in selected]


def _select_metadata_slots(
    scored_chunks: List[Tuple[float, Any]],
    top_k: int,
) -> List[Any]:
    """
    Metadata-aware slot allocation.

    Guarantees proportional representation of each (company, quarter, year)
    entity present in the candidate pool.

    Algorithm:
        1. Group candidates by entity.  Within each group, order is preserved
           from scored_chunks (i.e. best-scoring chunk for that entity first).
        2. Sort groups by their top chunk's global rank so the highest-quality
           entity gets priority if top_k < n_entities * base_slots.
        3. Each entity gets floor(top_k / n_entities) guaranteed slots.
        4. Remaining slots are filled by global score order.

    Args:
        scored_chunks: (relevance_score, chunk) pairs, sorted best-first.
        top_k:         Number of chunks to return.

    Returns:
        top_k chunks, sorted by relevance score (best first).
    """
    n = len(scored_chunks)
    top_k = min(top_k, n)
    if top_k == 0:
        return []

    # Group indices (into scored_chunks) by entity.
    # Since scored_chunks is sorted best-first, each group's indices are in
    # descending score order — the first index is that entity's best chunk.
    groups: Dict[Tuple, List[int]] = {}
    for idx, (_, chunk) in enumerate(scored_chunks):
        key = (
            getattr(chunk, "company", ""),
            getattr(chunk, "quarter", ""),
            getattr(chunk, "year", 0),
        )
        groups.setdefault(key, []).append(idx)

    n_entities = len(groups)
    base_slots = max(1, top_k // n_entities)

    # Sort groups by rank of their best chunk (lowest index = highest score first).
    sorted_groups = sorted(groups.items(), key=lambda kv: kv[1][0])

    selected_indices: List[int] = []
    used: set = set()

    # First pass — guaranteed base_slots per entity.
    for _key, indices in sorted_groups:
        for idx in indices[:base_slots]:
            if idx not in used and len(selected_indices) < top_k:
                used.add(idx)
                selected_indices.append(idx)

    # Second pass — fill remaining slots by global score order.
    for idx in range(n):
        if len(selected_indices) >= top_k:
            break
        if idx not in used:
            used.add(idx)
            selected_indices.append(idx)

    # Return in score-descending order.
    selected_indices.sort(key=lambda i: -scored_chunks[i][0])

    entity_counts: Dict[Tuple, int] = {}
    for idx in selected_indices:
        chunk = scored_chunks[idx][1]
        key = (
            getattr(chunk, "company", ""),
            getattr(chunk, "quarter", ""),
            getattr(chunk, "year", 0),
        )
        entity_counts[key] = entity_counts.get(key, 0) + 1

    logger.debug(
        "MetadataSlots: %d entities, base_slots=%d, coverage=%s",
        n_entities,
        base_slots,
        {f"{k[0]} {k[1]} {k[2]}": v for k, v in entity_counts.items()},
    )
    return [scored_chunks[i][1] for i in selected_indices[:top_k]]


def _select_source_cap(
    scored_chunks: List[Tuple[float, Any]],
    top_k: int,
    max_per_entity: Optional[int] = None,
) -> List[Any]:
    """
    Source diversity cap — per-entity upper bound on slot count.

    Greedily iterates through candidates (best score first) and selects
    each chunk unless its (company, quarter, year) entity has already
    consumed max_per_entity slots.  Deferred chunks fill any remaining
    slots after the greedy pass, so top_k is always met when the pool
    is large enough.

    Default cap: ceil(top_k / n_entities), minimum 2.  This naturally
    distributes slots evenly across filings without needing explicit
    configuration — e.g. top_k=6 with 4 filings → cap=2 per filing.
    Pass max_per_entity explicitly to override.

    Complements metadata_slots (which is a per-entity FLOOR).  Use
    source_cap when you want to prevent one filing from monopolising all
    slots; use metadata_slots when you want to guarantee a minimum.
    """
    n = len(scored_chunks)
    top_k = min(top_k, n)
    if top_k == 0:
        return []

    n_entities = len({
        (
            getattr(c, "company", ""),
            getattr(c, "quarter", ""),
            getattr(c, "year", 0),
        )
        for _, c in scored_chunks
    })

    cap = max_per_entity if max_per_entity is not None else max(2, math.ceil(top_k / max(n_entities, 1)))

    entity_counts: Dict[Tuple, int] = {}
    selected: List[Tuple[float, Any]] = []
    overflow: List[Tuple[float, Any]] = []

    for score, chunk in scored_chunks:
        key = (
            getattr(chunk, "company", ""),
            getattr(chunk, "quarter", ""),
            getattr(chunk, "year", 0),
        )
        if entity_counts.get(key, 0) < cap:
            selected.append((score, chunk))
            entity_counts[key] = entity_counts.get(key, 0) + 1
            if len(selected) == top_k:
                break
        else:
            overflow.append((score, chunk))

    # Fill remaining slots from overflow (best-scoring deferred chunks)
    for score, chunk in overflow:
        if len(selected) >= top_k:
            break
        selected.append((score, chunk))

    logger.debug(
        "SourceCap: cap=%d  n_entities=%d  coverage=%s",
        cap,
        n_entities,
        {f"{k[0]} {k[1]} {k[2]}": v for k, v in entity_counts.items()},
    )
    return [c for _, c in selected[:top_k]]


def _apply_diversity(
    scored_chunks: List[Tuple[float, Any]],
    top_k: int,
    diversity_mode: str,
    mmr_lambda: float,
    max_per_entity: Optional[int] = None,
) -> List[Any]:
    """Dispatch to the chosen diversity selection strategy."""
    if diversity_mode == "mmr":
        return _select_mmr(scored_chunks, top_k, mmr_lambda)
    if diversity_mode == "metadata_slots":
        return _select_metadata_slots(scored_chunks, top_k)
    if diversity_mode == "source_cap":
        return _select_source_cap(scored_chunks, top_k, max_per_entity)
    # "none" or anything else — plain top-k sort
    return [c for _, c in scored_chunks[:top_k]]


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
    diversity_mode: str = "none",
    mmr_lambda: float = 0.5,
    max_per_entity: Optional[int] = None,
) -> list:
    import json
    import openai

    if not chunks:
        return chunks

    passages = "\n\n".join(
        f"[{i + 1}] {c.text[:600]}"
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

        parsed = json.loads(raw)
        if isinstance(parsed, list):
            ranking = parsed
        elif isinstance(parsed, dict):
            ranking = next(
                (v for v in parsed.values() if isinstance(v, list)), None
            )
            if ranking is None:
                raise ValueError(f"No list found in LLM response: {parsed}")
        else:
            raise ValueError(f"Unexpected LLM response shape: {type(parsed)}")

        n = len(chunks)
        indices = [int(r) - 1 for r in ranking if 1 <= int(r) <= n]
        seen = set(indices)
        indices += [i for i in range(n) if i not in seen]

        # Convert LLM rank order to pseudo-scores: rank 0 → score 1.0, last → ~0.
        # This lets the diversity functions treat LLM-ranked candidates the same
        # way they treat CE-scored ones.
        reranked_all = [chunks[i] for i in indices]
        n_all = len(reranked_all)
        scored = [(1.0 - (rank / max(n_all, 1)), chunk)
                  for rank, chunk in enumerate(reranked_all)]

        logger.debug(
            "LLM re-ranker: %d candidates → diversity_mode=%s  order=%s",
            n, diversity_mode, [i + 1 for i in indices[:top_k]],
        )
        return _apply_diversity(scored, top_k, diversity_mode, mmr_lambda, max_per_entity)

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
    diversity_mode: str = "none",
    mmr_lambda: float = 0.5,
    max_per_entity: Optional[int] = None,
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
        raw_scores = encoder.predict(pairs)

        # Sort all candidates by CE score (best first) — diversity selection
        # sees the complete scored pool, not a pre-sliced top-k.
        scored = sorted(
            zip(raw_scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        logger.debug(
            "Cross-encoder: %d candidates scored  top_score=%.4f  diversity_mode=%s",
            len(chunks),
            scored[0][0] if scored else 0,
            diversity_mode,
        )
        return _apply_diversity(scored, top_k, diversity_mode, mmr_lambda, max_per_entity)

    except Exception as exc:
        logger.warning(
            "Cross-encoder re-ranker failed (%s) — returning original order.", exc
        )
        return chunks[:top_k]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(
    question: str,
    chunks: list,
    top_k: int,
    mode: Literal["llm", "cross_encoder"] = "llm",
    diversity_mode: Literal["none", "mmr", "metadata_slots", "source_cap"] = "none",
    mmr_lambda: float = 0.5,
    max_per_entity: Optional[int] = None,
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
        mode:                 ``"llm"``           — listwise LLM re-ranking.
                              ``"cross_encoder"`` — local sentence-transformers.
        diversity_mode:       Final selection strategy after scoring:
                              ``"none"``           — sort by score, take top_k
                                                    (default, current behaviour).
                              ``"mmr"``            — Maximal Marginal Relevance;
                                                    balances relevance vs. chunk
                                                    diversity (requires sklearn).
                              ``"metadata_slots"`` — guarantees proportional
                                                    slot coverage per
                                                    (company, quarter, year).
        mmr_lambda:           MMR relevance/diversity trade-off.
                              1.0 = pure relevance, 0.0 = pure diversity.
                              Only used when diversity_mode="mmr".
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

    if len(chunks) <= top_k and diversity_mode == "none":
        return chunks

    logger.info(
        "Re-ranking %d candidates → top %d  mode=%s  diversity=%s",
        len(chunks), top_k, mode, diversity_mode,
    )

    if mode == "cross_encoder":
        return _rerank_cross_encoder(
            question, chunks, top_k, cross_encoder_model, diversity_mode, mmr_lambda, max_per_entity
        )

    if not api_key:
        logger.warning(
            "rerank called with mode='llm' but no api_key — "
            "falling back to original order."
        )
        return chunks[:top_k]

    return _rerank_llm(
        question, chunks, top_k, api_key, model, diversity_mode, mmr_lambda, max_per_entity
    )
