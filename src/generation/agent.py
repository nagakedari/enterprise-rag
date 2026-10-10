"""
Deterministic tool-calling pipeline: route -> retrieve -> calculate -> synthesize.

    User question
        -> router.classify()          [1 cheap structured LLM call]
           decides: needs_retrieval? needs_calculation? which tool?
        -> deterministic Python:
             if needs_retrieval:  retrieve() / retrieve_per_filing()
        -> if needs_calculation:
             1 forced single-function LLM call fills in that tool's
             numeric arguments from the retrieved text (or the question
             itself, for tool_only questions) — it extracts, it does not
             compute
             -> FinancialAnalyticsTools.call(...) actually does the math,
                in plain Python, no LLM involved
        -> generator.generate() synthesizes the final cited answer from
           whatever context + calculation result the steps above produced
           [1 LLM call]

This intentionally is NOT an agentic loop: the control flow (which steps run,
in what order) is plain Python driven by one router decision, not re-decided
by the model turn by turn. Total LLM calls per question is fixed at 2 or 3
(router, optional argument-extraction, synthesis) instead of open-ended.

`answer_question()` below is the standalone pipeline (its own retrieval via
_execute_retrieval). Callers with their own already-configurable retrieval —
src/api/main.py's /chat endpoint and scripts/run_evaluation.py — use
`route_and_calculate()` instead: it applies just the classify-then-calculate
half of this pipeline on top of chunks they retrieved themselves, so the
identical calculation step (and identical "insufficient data" handling) runs
in all three places without being copy-pasted three times.
"""
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config import Config
from src.generation.generator import _build_context_block, _get_client, generate
from src.generation.router import RouterDecision, classify
from src.generation.tools import FinancialAnalyticsTools
from src.retrieval.retriever import RetrievedChunk, retrieve, retrieve_per_filing

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    answer: str
    sources: List[RetrievedChunk]
    path: str
    calculation: Optional[Dict[str, Any]] = None


def _execute_retrieval(decision: RouterDecision, config: Config) -> List[RetrievedChunk]:
    query = decision.retrieval_query or ""
    if decision.all_quarters:
        if not decision.company:
            raise ValueError("Router set all_quarters=true without a company ticker")
        return retrieve_per_filing(query, config, decision.company, chunks_per_filing=5)
    return retrieve(
        query, config, top_k=5,
        company=decision.company, year=decision.year, quarter=decision.quarter,
    )


def extract_calculation_arguments(
    tool_name: str,
    query: str,
    chunks: List[RetrievedChunk],
    config: Config,
) -> Dict[str, Any]:
    """
    Forced single-function tool call: the model can only call `tool_name`, so
    this step extracts that function's numeric arguments from the retrieved
    text (or the question itself, when there's no retrieval) — it never
    performs the calculation itself; FinancialAnalyticsTools.call() does.
    """
    tool_schema = next(t for t in FinancialAnalyticsTools.TOOLS if t["function"]["name"] == tool_name)

    if chunks:
        context = f"Context:\n\n{_build_context_block(chunks)}\n\nQuestion: {query}"
    else:
        context = f"Question: {query}"

    client = _get_client(config.router.api_key)
    response = client.chat.completions.create(
        model=config.router.model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Extract the exact arguments needed to call the `{tool_name}` "
                    "function from the context and question below. Use the precise "
                    "figures exactly as reported — do not round or estimate."
                ),
            },
            {"role": "user", "content": context},
        ],
        tools=[tool_schema],
        tool_choice={"type": "function", "function": {"name": tool_name}},
        temperature=config.router.temperature,
    )
    tool_call = response.choices[0].message.tool_calls[0]
    return json.loads(tool_call.function.arguments or "{}")


def run_calculation(
    decision: RouterDecision,
    query: str,
    chunks: List[RetrievedChunk],
    config: Config,
) -> Optional[Dict[str, Any]]:
    """
    Given an already-computed RouterDecision, extract arguments from `chunks`
    (or the question itself) and run the chosen FinancialAnalyticsTools
    function. Returns None when no calculation was needed, or when extraction
    determined the figures aren't available in `chunks` — the latter is
    logged at INFO as an expected outcome, not treated as a bug.
    """
    if not decision.needs_calculation:
        return None
    if not decision.calculation_tool:
        logger.warning("Router set needs_calculation=true without choosing a calculation_tool for %r", query)
        return None

    arguments = extract_calculation_arguments(decision.calculation_tool, query, chunks, config)
    try:
        calculation = FinancialAnalyticsTools.call(decision.calculation_tool, arguments)
        logger.info("Calculation %s(%s) -> %s", decision.calculation_tool, arguments, calculation)
        return calculation
    except (ValueError, TypeError) as exc:
        # The extraction call couldn't find a required figure in the
        # retrieved text (e.g. it came back null) — not a bug, just no
        # calculation to report. Caller falls back to plain generation.
        logger.info(
            "Calculation skipped — %s(%s) has missing/invalid arguments: %s",
            decision.calculation_tool, arguments, exc,
        )
        return None


def route_and_calculate(
    query: str,
    chunks: List[RetrievedChunk],
    config: Config,
) -> Optional[Dict[str, Any]]:
    """
    Classify whether `query` needs a computed value on top of already-
    retrieved `chunks`, and run it if so. For callers that already have their
    own configurable retrieval pipeline (src/api/main.py's /chat endpoint,
    scripts/run_evaluation.py) and only want the calculation half of the
    router's decision — see answer_question() below for the standalone
    pipeline that also acts on the router's retrieval decision.
    """
    decision = classify(query, config)
    return run_calculation(decision, query, chunks, config)


def answer_question(query: str, config: Config) -> AgentResult:
    """Route, then deterministically retrieve/calculate/synthesize (standalone pipeline)."""
    decision = classify(query, config)

    chunks: List[RetrievedChunk] = []
    if decision.needs_retrieval:
        chunks = _execute_retrieval(decision, config)

    calculation = run_calculation(decision, query, chunks, config)

    answer = generate(query, chunks, config.generation, calculation)
    return AgentResult(answer=answer, sources=chunks, path=decision.path, calculation=calculation)
