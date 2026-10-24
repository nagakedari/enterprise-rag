"""
Tool registry and executor for the agentic RAG loop (src/agentic/agent.py).

The agent sees eight tools:
  - plan_research           records the research plan; forced as the first
                            call of every run (see src/agentic/agent.py)
  - search_filings          retrieval over the 10-Q vector store
  - list_topics             inventory of every distinct topic in the
                            retrieved sources, required before answering an
                            overview / multi-part / multi-period question
  - calculate_change, analyze_trend, calculate_ratio,
    calculate_share_of_total, compare_across_entities
                            the deterministic calculators from
                            src/generation/tools.py (FinancialAnalyticsTools),
                            reused as-is — no LLM does arithmetic

Retrieval is injected, not implemented here: callers pass a `SearchFn` that
wraps whatever retrieval pipeline they already have configured (engine,
chunking strategy, hybrid alpha, reranking, per-filing, ...). That keeps this
package independent of how /chat or the evaluation CLI retrieve, and lets
both run the identical agent on top of their own retrieval settings.

Every tool failure (bad JSON, missing figure, unknown tool) is returned to
the model as an {"error": ...} payload instead of raised, so the model can
react — search again for a missing figure, or say what's missing — which is
the point of an agentic loop over a fixed pipeline. Retrieval exceptions are
the one thing NOT swallowed: an unreachable vector store is an
infrastructure failure, not "no data found", and should surface as such.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.generation.tools import FinancialAnalyticsTools
from src.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# search_fn(query, company=..., year=..., quarter=..., period_date=...) -> chunks
# period_date (datetime.date) selects the filing whose fiscal quarter contains
# it; year/quarter select a filing by its label.
SearchFn = Callable[..., List[RetrievedChunk]]

SEARCH_TOOL_NAME = "search_filings"
PLAN_TOOL_NAME = "plan_research"
TOPICS_TOOL_NAME = "list_topics"
_CALCULATOR_NAMES = {t["function"]["name"] for t in FinancialAnalyticsTools.TOOLS}

# Where in a 10-Q a sub-topic is discussed. Used to word the search query
# for that part of the filing, NOT as a metadata filter: ingested
# section_title values are too inconsistent to filter on (e.g. most AAPL
# chunks are labelled "Item 6. Exhibits"), so a hard filter would silently
# drop relevant passages.
FILING_PARTS = [
    "financial statements",
    "notes to financial statements",
    "MD&A",
    "market risk disclosures",
    "legal proceedings",
    "risk factors",
    "any",
]

_PLAN_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": PLAN_TOOL_NAME,
        "description": (
            "Record your research plan before searching: the sub-topics the "
            "answer needs, and for each one which companies, filing periods "
            "and part of the 10-Q to search, plus the search query to use. A "
            "single-fact question gets a one-item plan. A message that needs "
            "no filing data gets requires_filings=false and a direct_reply "
            "instead of sub-topics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "requires_filings": {
                    "type": "boolean",
                    "description": (
                        "False for greetings, thanks, small talk, questions about "
                        "this assistant, or anything that does not need SEC 10-Q "
                        "information. Nothing is searched when false."
                    ),
                },
                "direct_reply": {
                    "type": "string",
                    "description": (
                        "Only when requires_filings is false: the complete, brief "
                        "reply to the user (e.g. a greeting that says you answer "
                        "questions about the companies' SEC 10-Q filings). No "
                        "financial facts or figures."
                    ),
                },
                "sub_topics": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "What this part of the answer covers."},
                            "filing_part": {
                                "type": "string",
                                "enum": FILING_PARTS,
                                "description": "Part of the 10-Q where this topic is discussed.",
                            },
                            "search_query": {
                                "type": "string",
                                "description": (
                                    "Query worded the way that part of the filing "
                                    "talks about the topic, e.g. 'accrued legal "
                                    "liabilities claims contingencies' rather than 'legal risk'."
                                ),
                            },
                            "companies": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Uppercase tickers; empty for all companies.",
                            },
                            "periods": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Filings to search separately, as labels from "
                                    "the filing catalog (e.g. ['2023 Q1', '2023 Q2']); "
                                    "empty when the question is not period-specific."
                                ),
                            },
                        },
                        "required": ["topic", "filing_part", "search_query"],
                    },
                },
                "compare_across_periods": {
                    "type": "boolean",
                    "description": "True when the question asks how something changed or evolved over time.",
                },
            },
            "required": ["requires_filings", "compare_across_periods"],
        },
    },
}

_TOPICS_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOPICS_TOOL_NAME,
        "description": (
            "Inventory of the retrieved sources for an overview, multi-part or "
            "multi-period question. List every distinct subject the sources "
            "discuss that bears on the question — one entry per specific risk, "
            "metric, event or issue, at the granularity of the sources' own "
            "headings — including less prominent ones, not only the most "
            "repeated. These are NOT the plan's search areas. Give each its "
            "sources, the filings it appears in (repeats included) and how it "
            "changed across periods. Account for EVERY source: each [Source N] "
            "must appear under some topic or in not_relevant. The final answer "
            "must cover every entry."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topics": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string"},
                            "reported": {
                                "type": "boolean",
                                "description": (
                                    "False when the sources do not actually report this "
                                    "subject (e.g. a breakdown or metric the question asks "
                                    "for that the filing doesn't give)."
                                ),
                            },
                            "sources": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": (
                                    "[Source N] numbers that actually report this subject "
                                    "— empty when reported is false."
                                ),
                            },
                            "filings": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filing labels it appears in, e.g. ['MSFT 2022 Q3', 'MSFT 2023 Q1'].",
                            },
                            "change_over_time": {
                                "type": "string",
                                "description": (
                                    "How it changed across the filings: 'unchanged', "
                                    "'first appears in …', 'no longer mentioned after …', "
                                    "or which figures moved. Only what the sources show."
                                ),
                            },
                        },
                        "required": ["topic", "reported", "sources"],
                    },
                },
                "not_relevant": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "[Source N] numbers that do not bear on the question.",
                },
            },
            "required": ["topics", "not_relevant"],
        },
    },
}

_SEARCH_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": (
            "Search the SEC 10-Q filings and return the most relevant passages "
            "as numbered [Source N] blocks. Target one filing either by its "
            "label (year + quarter, exactly as in the filing catalog) or by a "
            "date inside its fiscal quarter (date) — never both. Call it "
            "multiple times to gather figures for different companies or periods."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Short search query naming the metric or topic, e.g. "
                        "'total net sales' or 'data center revenue'."
                    ),
                },
                "company": {
                    "type": "string",
                    "description": "Uppercase ticker symbol, e.g. 'AAPL'. Omit to search all companies.",
                },
                "year": {
                    "type": "integer",
                    "description": (
                        "Year of the filing LABEL, e.g. 2023 for the '2023 Q3' filing. "
                        "Use with quarter when the question names a filing like "
                        "'Q3 2023' or 'the Q3 2023 10-Q'."
                    ),
                },
                "quarter": {
                    "type": "string",
                    "enum": ["Q1", "Q2", "Q3", "Q4"],
                    "description": (
                        "Quarter of the filing LABEL. Labels are not calendar "
                        "quarters — check the filing catalog."
                    ),
                },
                "date": {
                    "type": "string",
                    "description": (
                        "YYYY-MM-DD. Returns only the filing whose fiscal quarter "
                        "contains this date. Use when the question gives a date or "
                        "describes a period ('quarter ended July 1, 2023', 'around "
                        "May 2023') instead of naming a filing."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

AGENT_TOOLS: List[Dict[str, Any]] = [_PLAN_TOOL, _SEARCH_TOOL, _TOPICS_TOOL, *FinancialAnalyticsTools.TOOLS]
# Offered during the search turns: plan and topic inventory are forced at
# fixed points by the loop (src/agentic/agent.py), not chosen by the model.
SEARCH_TURN_TOOLS: List[Dict[str, Any]] = [_SEARCH_TOOL, *FinancialAnalyticsTools.TOOLS]


@dataclass
class ToolCallRecord:
    """One tool invocation made by the agent, kept for tracing/debugging."""
    name: str
    arguments: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    source_ids: List[int] = field(default_factory=list)

    @property
    def is_calculation(self) -> bool:
        return self.name in _CALCULATOR_NAMES


# Near-duplicate collapsing. 10-Q text is largely repeated quarter to quarter
# (e.g. ~half of Microsoft's 2023 Q1 parent chunks are >=85% contained in
# its 2022 Q3 filing), so multi-period searches would otherwise return the
# same passages once per filing — inflating the context several-fold and
# leaving the model to notice the repetition itself. A passage is collapsed
# when >= _REPEAT_COVERAGE of its 5-word shingles already appeared in sources
# from another filing of the same company AND every figure in it appears
# there too, so period-specific figures are never collapsed. Figures are
# numbers with a comma, decimal point or 4+ digits — bare 1–3 digit integers
# are mostly page numbers, which differ between otherwise identical passages.
# 0.80 rather than higher because chunk boundaries shift between filings, so
# an unchanged paragraph often lands ~80–90% inside the earlier chunk.
_SHINGLE_WORDS = 5
_REPEAT_COVERAGE = 0.80
_MIN_CONTRIBUTION = 0.2
_WORD_RE = re.compile(r"[a-z]+|\d[\d,.]*\d|\d")
_NUMBER_RE = re.compile(r"\d[\d,.]*\d|\d")


def _figures(text: str) -> Set[str]:
    return {n for n in _NUMBER_RE.findall(text) if "," in n or "." in n or len(n) >= 4}


def _shingles(text: str) -> Set[Tuple[str, ...]]:
    words = _WORD_RE.findall(text.lower())
    n = _SHINGLE_WORDS
    return {tuple(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def _filing_label(chunk: RetrievedChunk) -> str:
    return f"{chunk.company} {chunk.year} {chunk.quarter}"


@dataclass
class SourceEntry:
    """How one retrieved chunk is presented to the agent."""
    kind: str                     # "new" | "seen" | "repeat"
    source_id: Optional[int]      # number for new/seen; None for a repeat
    chunk: RetrievedChunk
    repeat_of: List[int] = field(default_factory=list)   # earlier sources it repeats


class SourceRegistry:
    """
    Global [Source N] numbering across every search the agent makes.

    The answer's citations must refer to stable numbers, so a chunk keeps the
    number it was first given even if a later search returns it again, and
    numbering continues (rather than restarting at 1) across searches.
    Passages that repeat text already retrieved from another filing are not
    numbered: they are reported as a repeat of the earlier source(s).
    """

    def __init__(self) -> None:
        self._chunks: List[RetrievedChunk] = []
        self._ids: Dict[Tuple[str, int, str], int] = {}
        self._shingles: List[Set[Tuple[str, ...]]] = []
        self._numbers: List[Set[str]] = []
        # source_id -> filings whose text repeats it
        self.repeated_in: Dict[int, List[str]] = {}
        # Collapsed repeats, kept out of the prompt but returned to callers
        # so a claim like "unchanged in 2023 Q3" stays traceable to that
        # filing's text (and verifiable by the evaluation judge).
        self._repeats: List[Tuple[RetrievedChunk, List[int]]] = []
        self._repeat_keys: Set[Tuple[str, int, str]] = set()

    @property
    def repeats(self) -> List[Tuple[RetrievedChunk, List[int]]]:
        """(chunk, [Source N it repeats]) for every collapsed passage."""
        return list(self._repeats)

    @property
    def chunks(self) -> List[RetrievedChunk]:
        return list(self._chunks)

    def _find_repeat(self, chunk: RetrievedChunk, shingles, numbers) -> List[int]:
        """Earlier sources (other filings, same company) this chunk repeats, else []."""
        earlier = [
            i for i, c in enumerate(self._chunks)
            if c.company == chunk.company and c.source_file != chunk.source_file
        ]
        if not earlier or not shingles:
            return []
        covered = set().union(*(self._shingles[i] for i in earlier)) & shingles
        if len(covered) / len(shingles) < _REPEAT_COVERAGE:
            return []
        if not numbers <= set().union(*(self._numbers[i] for i in earlier)):
            return []
        overlaps = sorted(
            ((len(self._shingles[i] & shingles) / len(shingles), i) for i in earlier),
            reverse=True,
        )
        return [i + 1 for share, i in overlaps[:3] if share >= _MIN_CONTRIBUTION]

    def add(self, chunks: List[RetrievedChunk]) -> List[SourceEntry]:
        """Register chunks and decide how each is presented."""
        entries = []
        for chunk in chunks:
            key = (chunk.source_file, chunk.chunk_index, chunk.text)
            if key in self._ids:
                entries.append(SourceEntry("seen", self._ids[key], chunk))
                continue
            shingles = _shingles(chunk.text)
            numbers = _figures(chunk.text)
            repeat_of = self._find_repeat(chunk, shingles, numbers)
            if repeat_of:
                for source_id in repeat_of:
                    filings = self.repeated_in.setdefault(source_id, [])
                    if _filing_label(chunk) not in filings:
                        filings.append(_filing_label(chunk))
                if key not in self._repeat_keys:
                    self._repeat_keys.add(key)
                    self._repeats.append((chunk, repeat_of))
                entries.append(SourceEntry("repeat", None, chunk, repeat_of))
                continue
            self._chunks.append(chunk)
            self._shingles.append(shingles)
            self._numbers.append(numbers)
            self._ids[key] = len(self._chunks)
            entries.append(SourceEntry("new", self._ids[key], chunk))
        return entries


def _format_sources(entries: List[SourceEntry]) -> str:
    parts: List[str] = []
    for entry in entries:
        chunk = entry.chunk
        period = f" (quarter ended {chunk.period_end:%b %d, %Y})" if chunk.period_end else ""
        filing = f"{_filing_label(chunk)} filing{period} | {chunk.source_file} | p.{chunk.page_start}–{chunk.page_end}"
        if entry.kind == "repeat":
            refs = "".join(f"[Source {i}]" for i in entry.repeat_of)
            parts.append(
                f"[{filing}]\nRepeats {refs} — this filing contains the same text "
                f"(no new figures). Cite {refs} and note it was reported again in "
                f"{_filing_label(chunk)}."
            )
            continue
        header = f"[Source {entry.source_id}: {filing}]"
        body = chunk.text.strip() if entry.kind == "new" else "(already retrieved above — same text)"
        parts.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(parts)


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_date(value: Any) -> Optional[date]:
    """Parse YYYY-MM-DD; raises ValueError for anything else non-empty."""
    if value in (None, ""):
        return None
    return date.fromisoformat(str(value).strip()[:10])


def _optional_upper(value: Any) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    return value.strip().upper() or None


class AgentToolExecutor:
    """Executes the agent's tool calls and keeps the source/call history."""

    def __init__(self, search_fn: SearchFn) -> None:
        self._search_fn = search_fn
        self.sources = SourceRegistry()
        self.records: List[ToolCallRecord] = []
        self.plan: Optional[Dict[str, Any]] = None
        self.topics: Optional[List[Dict[str, Any]]] = None

    def execute(self, name: str, raw_arguments: Optional[str]) -> str:
        """Run one tool call; returns the content for the `tool` message."""
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            return self._error(name, {}, f"arguments were not valid JSON: {exc}")

        if name == PLAN_TOOL_NAME:
            return self._record_plan(arguments)
        if name == TOPICS_TOOL_NAME:
            return self._record_topics(arguments)
        if name == SEARCH_TOOL_NAME:
            return self._search(arguments)

        try:
            result = FinancialAnalyticsTools.call(name, arguments)
        except KeyError:
            return self._error(name, arguments, f"unknown tool {name!r}")
        except (ValueError, TypeError) as exc:
            # Typically a required figure came back null/missing — let the
            # model search for it or report it as unavailable.
            return self._error(name, arguments, str(exc))

        logger.info("Agent tool %s(%s) -> %s", name, arguments, result)
        self.records.append(ToolCallRecord(name=name, arguments=arguments, result=result))
        return json.dumps(result)

    def _record_plan(self, arguments: Dict[str, Any]) -> str:
        if arguments.get("requires_filings") is False:
            # Greeting / small talk / off-topic: answered from the plan's
            # direct_reply without any retrieval (see agent.run_agent).
            self.plan = arguments
            logger.info("Agent plan: no filings needed — direct reply")
            self.records.append(ToolCallRecord(name=PLAN_TOOL_NAME, arguments=arguments, result={"recorded": True}))
            return "No filing search needed."

        sub_topics = arguments.get("sub_topics")
        if not isinstance(sub_topics, list) or not sub_topics:
            return self._error(PLAN_TOOL_NAME, arguments, "sub_topics must be a non-empty list")

        # A later call replaces the plan (the model may refine it mid-run).
        self.plan = arguments
        logger.info("Agent plan: %s", json.dumps(arguments))
        self.records.append(ToolCallRecord(name=PLAN_TOOL_NAME, arguments=arguments, result={"recorded": True}))
        return (
            "Plan recorded. Execute it now: run search_filings for every sub-topic "
            "(and separately for each listed period), issuing independent searches "
            "together in one turn. Before answering, check every sub-topic and "
            "period is covered or name it as not found."
        )

    def _record_topics(self, arguments: Dict[str, Any]) -> str:
        topics = arguments.get("topics")
        if not isinstance(topics, list) or not topics:
            return self._error(TOPICS_TOOL_NAME, arguments, "topics must be a non-empty list")

        # Every source must be accounted for, so content outside the most
        # prominent passages can't be skipped silently.
        accounted = {i for t in topics for i in (t.get("sources") or []) if isinstance(i, int)}
        accounted |= {i for i in (arguments.get("not_relevant") or []) if isinstance(i, int)}
        missing = [i for i in range(1, len(self.sources.chunks) + 1) if i not in accounted]
        if missing:
            refs = ", ".join(f"[Source {i}]" for i in missing)
            return self._error(
                TOPICS_TOOL_NAME, arguments,
                f"{refs} not accounted for. Re-read them and call list_topics again "
                "with each one under a topic it supports (add topics as needed) or in not_relevant.",
            )

        self.topics = topics
        logger.info("Agent topics: %s", json.dumps(topics))
        self.records.append(ToolCallRecord(name=TOPICS_TOOL_NAME, arguments=arguments, result={"recorded": len(topics)}))
        return (
            f"{len(topics)} topics recorded. Write the final answer now: cover every "
            "topic listed — state topics with reported=false as not reported in the "
            "filings, with no substitute figures — cite each topic's sources, and "
            "describe change across periods from change_over_time."
        )

    def _search(self, arguments: Dict[str, Any]) -> str:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return self._error(SEARCH_TOOL_NAME, arguments, "query must be a non-empty string")

        company = _optional_upper(arguments.get("company"))
        year = _optional_int(arguments.get("year"))
        quarter = _optional_upper(arguments.get("quarter"))
        try:
            period_date = _optional_date(arguments.get("date"))
        except ValueError:
            return self._error(SEARCH_TOOL_NAME, arguments, "date must be YYYY-MM-DD")
        if period_date and (year or quarter):
            # A label and a date can point at different filings; the date wins
            # since it is unambiguous.
            year = quarter = None

        chunks = self._search_fn(query, company=company, year=year, quarter=quarter, period_date=period_date)
        entries = self.sources.add(chunks)
        source_ids = [e.source_id for e in entries if e.source_id is not None]
        repeats = sorted({i for e in entries for i in e.repeat_of})

        logger.info(
            "Agent search %r (company=%s year=%s quarter=%s date=%s) -> sources %s, repeats of %s",
            query, company, year, quarter, period_date, source_ids, repeats,
        )
        self.records.append(ToolCallRecord(
            name=SEARCH_TOOL_NAME,
            arguments=arguments,
            result={"sources": source_ids, "repeats_of": repeats},
            source_ids=source_ids,
        ))
        if not entries:
            return "No matching passages found. Try a broader query or fewer filters."
        return _format_sources(entries)

    def _error(self, name: str, arguments: Dict[str, Any], message: str) -> str:
        logger.info("Agent tool %s(%s) failed: %s", name, arguments, message)
        self.records.append(ToolCallRecord(name=name, arguments=arguments, error=message))
        return json.dumps({"error": message})
