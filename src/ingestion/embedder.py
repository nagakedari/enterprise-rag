"""
Embedding module – wraps the OpenAI Embeddings API.

Phase-1 note:
    We call the API directly and pass vectors into Weaviate.
    This gives maximum transparency for learning/debugging.
    Phase 2 can delegate vectorisation to Weaviate's text2vec-openai
    module to remove the extra round-trip.
"""
import logging
from typing import List

import openai

from src.config import EmbeddingConfig

logger = logging.getLogger(__name__)


def create_embeddings(texts: List[str], config: EmbeddingConfig) -> List[List[float]]:
    """
    Return one embedding vector per input text.

    Batches requests to respect API limits.
    Raises openai.OpenAIError on failure (let the caller decide to retry).
    """
    if not config.api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )

    client = openai.OpenAI(api_key=config.api_key)
    all_embeddings: List[List[float]] = []
    total = len(texts)

    for batch_start in range(0, total, config.api_batch_size):
        batch = texts[batch_start : batch_start + config.api_batch_size]
        batch_end = batch_start + len(batch)
        logger.info(
            "Embedding batch %d–%d / %d  model=%s",
            batch_start + 1, batch_end, total, config.model,
        )
        response = client.embeddings.create(model=config.model, input=batch)
        # OpenAI guarantees ordering matches input
        batch_vecs = [item.embedding for item in response.data]
        all_embeddings.extend(batch_vecs)

    logger.info("Created %d embeddings (dim=%d)", len(all_embeddings), config.dimensions)
    return all_embeddings
