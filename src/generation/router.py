"""
Lightweight routing classifier for the deterministic tool-calling pipeline.

A single structured LLM call decides, for a given question:
  - whether retrieval from the vector DB is needed (and with what query/filters)
  - whether a calculation is needed, and if so exactly which
    FinancialAnalyticsTools function applies

Everything downstream of this decision (see src/generation/agent.py) runs as
plain deterministic Python: retrieval calls retrieve()/retrieve_per_filing()
directly, and the calculation itself is a normal function call — no LLM does
arithmetic. The only other LLM involvement in the pipeline is filling in the
chosen calculation tool's numeric arguments once source text is in hand, and
the final answer synthesis. That keeps the number of LLM calls per question
fixed at 2 or 3, rather than an open-ended agentic loop.
"""
import json
import logging
from dataclasses import dataclass
from typing import Optional

from src.config import Config
from src.generation.generator import _get_client

logger = logging.getLogger(__name__)

_ROUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "route",
        "description": (
            "Classify how to answer a question about SEC 10-Q filings: "
            "whether retrieval from the vector DB is needed, whether a "
            "calculation is needed, and which one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "enum": ["rag_only", "tool_only", "rag_and_tool"],
                },
                "needs_retrieval": {"type": "boolean"},
                "retrieval_query": {
                    "type": "string",
                    "description": (
                        "Short search query for the vector DB, e.g. "
                        "'gross margin cost of revenue'. Required if needs_retrieval."
                    ),
                },
                "company": {
                    "type": "string",
                    "description": "Ticker symbol, e.g. 'AAPL'. Omit if not company-specific.",
                },
                "year": {"type": "integer"},
                "quarter": {
                    "type": "string",
                    "description": "e.g. 'Q2'. Omit if the question spans multiple quarters.",
                },
                "all_quarters": {
                    "type": "boolean",
                    "description": (
                        "True for trend/time-series questions across quarters with "
                        "no single quarter specified. Requires company."
                    ),
                },
                "needs_calculation": {"type": "boolean"},
                "calculation_tool": {
                    "type": "string",
                    "enum": [
                        "calculate_change",
                        "analyze_trend",
                        "calculate_ratio",
                        "calculate_share_of_total",
                        "compare_across_entities",
                    ],
                    "description": (
                        "calculate_change: one before/after value pair. "
                        "analyze_trend: 3+ chronological values of the same metric. "
                        "calculate_ratio: a margin/rate/generic a-over-b ratio for one period. "
                        "calculate_share_of_total: a part's percentage of a whole in one period. "
                        "compare_across_entities: ranking/diffing a metric across 2+ companies "
                        "or labeled periods. Required if needs_calculation."
                    ),
                },
            },
            "required": ["path", "needs_retrieval", "needs_calculation"],
        },
    },
}

_SYSTEM_PROMPT = """You are a routing classifier for a financial-analyst RAG system over SEC 10-Q filings.

For the user's question, decide:
1. needs_retrieval — false only if the question supplies every number it needs itself; true otherwise.
2. needs_calculation — true for any change/trend/margin/tax-rate/share-of-total/comparison question; false for plain fact lookup or qualitative summary.
3. If needs_calculation, calculation_tool — exactly one of calculate_change, analyze_trend, calculate_ratio, calculate_share_of_total, compare_across_entities.
4. If needs_retrieval, retrieval_query (a short natural-language search query for the metric) and any of company/year/quarter you can infer from the question. Set all_quarters=true for trend/time-series questions spanning quarters with no single quarter specified.

Set path consistently with needs_retrieval/needs_calculation:
  rag_only     -> needs_retrieval=true,  needs_calculation=false
  tool_only    -> needs_retrieval=false, needs_calculation=true
  rag_and_tool -> needs_retrieval=true,  needs_calculation=true

Always call the route tool."""


@dataclass
class RouterDecision:
    path: str
    needs_retrieval: bool
    needs_calculation: bool
    retrieval_query: Optional[str] = None
    company: Optional[str] = None
    year: Optional[int] = None
    quarter: Optional[str] = None
    all_quarters: bool = False
    calculation_tool: Optional[str] = None


def classify(query: str, config: Config) -> RouterDecision:
    """Single structured LLM call that decides the execution path for `query`."""
    client = _get_client(config.router.api_key)
    response = client.chat.completions.create(
        model=config.router.model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        tools=[_ROUTE_TOOL],
        tool_choice={"type": "function", "function": {"name": "route"}},
        temperature=config.router.temperature,
    )
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments or "{}")

    decision = RouterDecision(
        path=args["path"],
        needs_retrieval=args["needs_retrieval"],
        needs_calculation=args["needs_calculation"],
        retrieval_query=args.get("retrieval_query"),
        company=args.get("company"),
        year=args.get("year"),
        quarter=args.get("quarter"),
        all_quarters=args.get("all_quarters", False),
        calculation_tool=args.get("calculation_tool"),
    )
    logger.info("Router classified %r -> %s", query, decision)
    return decision
