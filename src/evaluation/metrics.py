"""
Non-LLM evaluation metrics derived from golden Q&A pairs.

  exactness(generated, golden)
      Token-level F1 between the generated and golden answer.
      No API calls required.

  answer_similarity(generated, golden, embedding_config)
      Cosine similarity between OpenAI embeddings of the two answers.
      Requires one OpenAI API call (2 texts per call).

  geval_context_relevance(question, contexts, api_key, model)
      GEval-style LLM scoring: asks the LLM to rate how relevant
      the retrieved contexts are to the question on a [0, 1] scale.

  geval_factual_error_rate(question, generated, golden, api_key, model)
      GEval-style LLM scoring: decomposes the generated answer into atomic
      claims and checks what fraction are contradicted by / absent from the
      golden answer.  Returns a hallucination rate in [0, 1] where 1.0 means
      every claim is factually wrong.
"""
import logging
import re
from collections import Counter
from typing import List

import numpy as np

from src.config import EmbeddingConfig
from src.ingestion.embedder import create_embeddings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase and extract alphanumeric tokens."""
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# Metric 1 – Exactness (token F1, no LLM)
# ---------------------------------------------------------------------------

def exactness(generated: str, golden: str) -> float:
    """
    Token-level F1 between *generated* and *golden* answer.

    Uses bag-of-words (Counter) token overlap — the standard SQuAD F1 approach.
    Unlike set-based F1, repeated tokens count proportionally, so an answer that
    mentions "revenue" five times and one that mentions it once are distinguished.

    Returns a float in [0, 1].
    """
    gen_counts = Counter(_tokenize(generated))
    gold_counts = Counter(_tokenize(golden))

    if not gen_counts or not gold_counts:
        return 0.0

    # Intersection: for each token, take the min count in both answers
    num_common = sum(
        min(gen_counts[tok], gold_counts[tok]) for tok in gen_counts if tok in gold_counts
    )

    if num_common == 0:
        return 0.0

    precision = num_common / sum(gen_counts.values())
    recall = num_common / sum(gold_counts.values())
    f1 = 2 * precision * recall / (precision + recall)
    return round(float(f1), 4)


# ---------------------------------------------------------------------------
# Metric 2 – Answer Similarity (embedding cosine, no extra LLM)
# ---------------------------------------------------------------------------

def answer_similarity(
    generated: str,
    golden: str,
    embedding_config: EmbeddingConfig,
) -> float:
    """
    Cosine similarity between OpenAI embeddings of *generated* and *golden*.

    Returns a float in [0, 1].  Values closer to 1.0 indicate semantically
    similar answers even when surface wording differs.
    """
    vectors = create_embeddings([generated, golden], embedding_config)
    gen_vec = np.array(vectors[0], dtype=np.float32)
    gold_vec = np.array(vectors[1], dtype=np.float32)

    norm_gen = np.linalg.norm(gen_vec)
    norm_gold = np.linalg.norm(gold_vec)

    if norm_gen == 0 or norm_gold == 0:
        return 0.0

    cos_sim = float(np.dot(gen_vec, gold_vec) / (norm_gen * norm_gold))
    # Cosine can be slightly negative for dissimilar texts; clamp to [0, 1]
    return round(max(0.0, cos_sim), 4)


# ---------------------------------------------------------------------------
# GEval helper – Context Relevance (LLM scoring, no reference required)
# ---------------------------------------------------------------------------

_CONTEXT_RELEVANCE_PROMPT = """\
You are an impartial evaluator of information retrieval quality.

Your task: rate how relevant the retrieved context passages are for \
answering the given question about SEC 10-Q financial filings.

Question:
{question}

Retrieved Context:
{context}

Scoring rubric:
  1.0 - All passages are directly relevant and sufficient to answer the question
  0.75 - Most passages are relevant; minor irrelevant material present
  0.5  - Roughly half the content is relevant; answer may be partially supported
  0.25 - Only a small portion is relevant; question mostly unanswerable from context
  0.0  - None of the passages are relevant to the question

Respond with a SINGLE decimal number between 0.0 and 1.0. No other text."""


def geval_context_relevance(
    question: str,
    contexts: List[str],
    api_key: str,
    model: str = "gpt-4o-mini",
    max_contexts: int = 5,
) -> float:
    """
    GEval-style LLM scoring of context relevance.

    Sends the question and (up to *max_contexts*) retrieved passages to the
    LLM and asks it to return a relevance score in [0, 1].

    Args:
        question:     The user's original question.
        contexts:     List of retrieved context strings (ordered by score).
        api_key:      OpenAI API key.
        model:        LLM to use for scoring (default: gpt-4o-mini).
        max_contexts: Cap to avoid exceeding context-window limits.

    Returns:
        A float in [0, 1].
    """
    import openai  # local import – only needed when this function is called

    context_block = "\n\n---\n\n".join(
        f"[Passage {i + 1}]\n{ctx.strip()}"
        for i, ctx in enumerate(contexts[:max_contexts])
    )

    prompt = _CONTEXT_RELEVANCE_PROMPT.format(
        question=question,
        context=context_block,
    )

    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (response.choices[0].message.content or "").strip()
        score = float(raw)
        return round(max(0.0, min(1.0, score)), 4)
    except Exception as exc:
        logger.warning("geval_context_relevance failed: %s", exc)
        return float("nan")


# ---------------------------------------------------------------------------
# GEval helper – Factual Error Rate (hallucination vs golden answer)
# ---------------------------------------------------------------------------

_FACTUAL_ERROR_PROMPT = """\
You are a factual accuracy auditor for financial Q&A systems.

Your task: identify what fraction of the claims in the GENERATED ANSWER are
factually incorrect or directly contradicted by the REFERENCE ANSWER for a
question about SEC 10-Q filings.

Question:
{question}

Reference Answer (ground truth):
{golden}

Generated Answer (to audit):
{generated}

Instructions:
1. Break the Generated Answer into individual factual claims (numbers, dates,
   company names, financial figures, trends, comparisons, etc.).
2. For each claim, decide: is it supported, contradicted, or not verifiable
   from the Reference Answer?
3. Count the claims that are CONTRADICTED or that assert specific facts absent
   from the reference (i.e. potentially hallucinated).
4. Return: contradicted_or_hallucinated_claims / total_claims

Scoring rubric:
  0.0  - Every claim in the generated answer is supported by the reference
  0.25 - A small fraction of claims are wrong or unverifiable
  0.5  - Roughly half the claims are wrong or unverifiable
  0.75 - Most claims are wrong or unverifiable
  1.0  - No claims are supported by the reference (complete hallucination)

Respond with a SINGLE decimal number between 0.0 and 1.0. No other text."""


def geval_factual_error_rate(
    question: str,
    generated: str,
    golden: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> float:
    """
    GEval-style LLM measurement of factual hallucination rate per answer.

    Decomposes the generated answer into atomic claims and asks the LLM
    what fraction are contradicted by or absent from the golden answer.

    Args:
        question:  The original user question.
        generated: The RAG system's generated answer.
        golden:    The ground-truth reference answer.
        api_key:   OpenAI API key.
        model:     LLM used for scoring (default: gpt-4o-mini).

    Returns:
        A float in [0, 1].
          0.0 → no hallucination detected (all claims supported)
          1.0 → complete hallucination (no claims supported)
    """
    import openai  # local import

    prompt = _FACTUAL_ERROR_PROMPT.format(
        question=question,
        golden=golden.strip(),
        generated=generated.strip(),
    )

    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (response.choices[0].message.content or "").strip()
        score = float(raw)
        return round(max(0.0, min(1.0, score)), 4)
    except Exception as exc:
        logger.warning("geval_factual_error_rate failed: %s", exc)
        return float("nan")
