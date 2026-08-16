"""
Shared run-tag construction for evaluation outputs.

Both scripts/run_evaluation.py (CLI) and src/api/evaluation_service.py (web
backend) must build the exact same filename tag for a given set of parameters
so that "has this configuration already been evaluated?" lookups agree with
what the CLI actually writes to disk. This module is the single source of
truth for that algorithm.

Tag segments are OMITTED (not defaulted) when a parameter is None — the tag is
a function of which flags were explicitly passed, not of the resolved config
value. That asymmetry is intentional (it mirrors historical CLI usage) and
must be preserved by every caller.
"""
from typing import Optional


def resolve_chunking_strategy(chunking_strategy: Optional[str], use_smart: bool) -> str:
    """Resolve the effective chunking strategy, mirroring the CLI's --use-smart fallback."""
    return chunking_strategy or ("parent_child" if use_smart else "basic")


_QTYPE_ABBREV = {
    "Multi-Doc RAG":                "multidoc",
    "Single-Doc Multi-Chunk RAG":   "sdmulti",
    "Single-Doc Single-Chunk RAG":  "sdsingle",
}


_DIVERSITY_ABBREV = {
    "mmr":            "mmr",
    "metadata_slots": "metaslots",
    "source_cap":     "sourcecap",
}


def build_run_tag(
    engine: str,
    chunking_strategy: str,
    filter_mode: str,
    config_retrieval_mode: str,
    retrieval_mode: Optional[str] = None,
    alpha: Optional[float] = None,
    top_k: Optional[int] = None,
    rerank_mode: Optional[str] = None,
    rerank_top_k: Optional[int] = None,
    diversity_mode: Optional[str] = None,
    mmr_lambda: Optional[float] = None,
    per_filing: bool = False,
    chunks_per_filing: Optional[int] = None,
    max_per_entity: Optional[int] = None,
    question_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Build the run tag embedded in evaluation_results_<run_tag>.csv filenames.

    Args:
        engine:               "custom" or "llamaindex".
        chunking_strategy:    Already-resolved via resolve_chunking_strategy()
                              — never None here.
        filter_mode:          "llm" or "regex".
        config_retrieval_mode: Fallback retrieval mode (config.retrieval.mode)
                              used when retrieval_mode is None.
        retrieval_mode:       Explicit override, or None to use the config default.
        alpha:                Hybrid BM25/vector balance, or None if not passed.
        top_k:                Chunks retrieved per question, or None if not passed.
        rerank_mode:          "llm" / "cross_encoder", or None if reranking is off.
        rerank_top_k:         Chunks kept after reranking, or None.
        diversity_mode:       "none" / "mmr" / "metadata_slots", or None.
                              Omitted from tag when "none" or not provided so
                              existing run filenames are unaffected.
        mmr_lambda:           MMR lambda (only appended when diversity_mode="mmr"
                              and lambda differs from the default 0.5).
        question_type:        Q&A question-type filter string, or None (all types).
                              Encoded as a short abbreviation so filenames stay readable.
        seed:                 Random seed used for Q&A sampling.  Only appended when
                              it differs from the default (42) so existing filenames
                              are unaffected.

    Returns:
        The run tag string, e.g.
        "custom_parent_child_hybrid_a0.25_k10_llmfilters_cross_encoderrerank_rt6_mmr"
        "custom_parent_child_hybrid_a0.25_k10_llmfilters_cross_encoderrerank_rt6_metaslots"
    """
    mode_tag = retrieval_mode or config_retrieval_mode
    alpha_tag = f"_a{alpha}" if alpha is not None else ""
    topk_tag = f"_k{top_k}" if top_k is not None else ""
    rerank_tag = f"_{rerank_mode}rerank" if rerank_mode else ""
    rerank_topk_tag = f"_rt{rerank_top_k}" if (rerank_mode and rerank_top_k is not None) else ""
    # Encode diversity mode when non-default so different selection strategies
    # produce distinct filenames and never overwrite each other.
    effective_diversity = diversity_mode if diversity_mode and diversity_mode != "none" else None
    diversity_tag = f"_{_DIVERSITY_ABBREV.get(effective_diversity, effective_diversity)}" \
        if effective_diversity else ""
    # Append MMR lambda only when it's non-default (0.5) to keep filenames short.
    lambda_tag = (
        f"_l{mmr_lambda}"
        if (effective_diversity == "mmr" and mmr_lambda is not None and mmr_lambda != 0.5)
        else ""
    )
    # Per-filing retrieval tag: "pf" when enabled, "pf_N" when chunks_per_filing != 3.
    per_filing_tag = ""
    if per_filing:
        per_filing_tag = "_pf"
        if chunks_per_filing is not None and chunks_per_filing != 3:
            per_filing_tag += f"{chunks_per_filing}"
    # max_per_entity only matters when source_cap is active.
    max_entity_tag = f"_cap{max_per_entity}" if (max_per_entity is not None and effective_diversity == "source_cap") else ""
    # Encode question_type filter so runs on different subsets never share a filename.
    qtype_tag = f"_{_QTYPE_ABBREV.get(question_type, 'qt')}" if question_type else ""
    # Encode seed only when non-default so existing filenames are not renamed.
    seed_tag = f"_s{seed}" if (seed is not None and seed != 42) else ""
    return (
        f"{engine}_{chunking_strategy}_{mode_tag}{alpha_tag}{topk_tag}"
        f"_{filter_mode}filters{rerank_tag}{rerank_topk_tag}"
        f"{diversity_tag}{lambda_tag}{max_entity_tag}{per_filing_tag}{qtype_tag}{seed_tag}"
    )
