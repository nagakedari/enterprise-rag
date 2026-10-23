"""
Agentic RAG: the model itself decides when to retrieve, what to calculate,
and when it has enough to answer.

    User question
        -> first LLM turn on the stronger planner model (AGENT_PLANNER_MODEL),
           forced via tool_choice to call plan_research: a structured plan of
           sub-topics, each with the part of the 10-Q it lives in, a search
           query, companies and periods (a one-item plan for single-fact
           questions). Enforced by the API, not requested in the prompt —
           prompt-only planning was skipped even by gpt-4o. Costs one extra
           LLM turn per question.
        -> later LLM turns (AGENT_MODEL) with tools=[search_filings, <5 calculators>], tool_choice="auto"
             -> model calls search_filings (any number of times, any filters)
             -> model calls a calculator with figures copied from the sources
             -> tool results are appended as `tool` messages
        -> repeat until the model replies without a tool call (the final,
           cited answer) or max_iterations is reached, after which one last
           turn with tool_choice="none" forces an answer from what was gathered

Compared to the deterministic pipeline in src/generation/agent.py (router ->
retrieve -> extract args -> calculate -> synthesize, fixed 2-3 LLM calls),
nothing here is decided up front: the model can issue follow-up searches when
a figure is missing, search each quarter separately for a trend, call more
than one calculator, or skip tools entirely. The cost is an open-ended (but
capped) number of LLM calls per question.

Arithmetic still never happens in the model — calculators are the same
FinancialAnalyticsTools functions, run in Python by AgentToolExecutor.

The answer-formatting rules (verbatim figures, [Source N] citations, temporal
anchoring, ...) are the generator's _SYSTEM_PROMPT, reused unchanged so
agentic and plain answers are held to the same standard and scored the same
way by the evaluation metrics.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.agentic.tools import (
    AGENT_TOOLS, PLAN_TOOL_NAME, SEARCH_TURN_TOOLS, TOPICS_TOOL_NAME, AgentToolExecutor, SearchFn,
    ToolCallRecord,
)
from src.config import AgentConfig
from src.generation.generator import _SYSTEM_PROMPT as _ANSWER_RULES
from src.generation.generator import _get_client
from src.retrieval.retriever import FilingPeriod, RetrievedChunk

logger = logging.getLogger(__name__)

_AGENT_INSTRUCTIONS = """

AGENT MODE — you have tools and decide yourself how to use them:

- Your first action is always plan_research. For a single-fact question the \
plan is one sub-topic. For a broad, multi-part or multi-period question, list \
every sub-topic the answer needs. A 10-Q discusses the same subject in \
several places — e.g. risks appear in risk factors, but also in MD&A \
(industry and economic conditions), market risk disclosures, and the notes \
to the financial statements (contingencies, litigation, income taxes) — so \
give each sub-topic the filing_part it lives in and a search_query worded \
the way that part of the filing talks about it. List periods separately \
when the question spans time.
- Then execute the plan with search_filings, one search per sub-topic and \
period. Do not restate the plan in the final answer.
- PERIODS. Map every period in the question to filings in the FILING CATALOG \
given with the question (label -> fiscal quarter it covers). Labels are not \
calendar quarters and differ by company. If the question names a filing \
("Q2 2023", "the Q3 2022 10-Q"), search with that label's year and quarter. \
If it gives a date or describes a time ("quarter ended July 1, 2023", "around \
May 2023"), search with date set to a day inside the matching filing's range. \
Plan and search only filings listed in the catalog; if a period the question \
asks about is not covered, say so instead of searching for it.
- Before using a figure, check that its column date ("Three Months Ended \
...") matches the quarter end of the period asked about — 10-Q tables also \
show prior-year and nine-month columns.
- search_filings retrieves passages from the 10-Q filings. Call it whenever you \
need facts or figures — never answer from general knowledge. Pass company/year/\
quarter filters when the question names them. For questions spanning several \
periods or companies (trends, changes over time, comparisons), make sure you \
have the figure for EVERY period/company involved — search each one separately \
if a single search does not return them all. Results arrive as [Source N] \
blocks numbered globally across all your searches; cite those numbers.
- calculate_change, analyze_trend, calculate_ratio, calculate_share_of_total and \
compare_across_entities are Python calculators. NEVER do arithmetic yourself: \
whenever the answer needs a change, trend, ratio, margin, rate, share or \
comparison, first find the exact input figures with search_filings, then call \
the matching calculator with those figures copied exactly as reported. A \
calculator's output is the [Calculation Result] referred to in rule 7.
- If a tool returns an error (e.g. a missing figure), search again for what is \
missing, or state in your answer exactly what could not be found.
- Independent tool calls can be issued together in a single turn.
- CHANGE OVER TIME. When the question asks how something changed, evolved or \
trended, gather the same information from each filing period and report what \
stayed the same, what was added or dropped, and which reported figures moved. \
Comparing what the sources from different periods say is grounded synthesis, \
not the inference that rule 4 forbids — cite the sources for each period.
- A search result may say a passage "Repeats [Source N]": that filing contains \
the same text as the earlier source — evidence the topic was reported \
unchanged in that period. Cite [Source N] for it.
- Before answering, check the plan: every sub-topic and every period it lists \
should be covered by a search result, or named in the answer as not found. \
Search again for any gap.
- ANSWER SCOPE. Lead with the direct answer to exactly what was asked, and \
stop there: do not add figures, ratios, comparisons or prior-period context \
the question did not ask for (for a change-over-time question, the periods \
it covers ARE what was asked). If the filings do not contain what the \
question asks for — e.g. a breakdown, comparison or metric they don't \
report — say so in the first sentence, then briefly note what related \
information they do report. Never assemble a substitute answer from nearby \
figures as if it answered the question.
- MISSING INFORMATION OVERRIDES RULES 4 AND 6. When the filings do not report \
something the question asks for, say so plainly and stop on that point: rule \
4's "Based on available sources" continuation may only give information the \
question itself asks for, never a stand-in (e.g. total revenue for a revenue \
category that is not reported). Rule 6's comparison applies only when every \
compared figure is reported; if one side is missing, say the comparison \
cannot be made instead of comparing substitutes.
- PERIOD NAMING. In the answer, refer to every period by its quarter-end \
date ("the quarter ended July 1, 2023"), taken from the filing catalog or \
the column header; you may add the filing label in parentheses. Never \
present a filing label ("Q3 2023") on its own as the name of a period — \
even where the sources or your topic notes use labels like that.
- For an overview, multi-part or multi-period question, when you have \
finished searching reply only with READY. You will then list the topics the \
sources cover (list_topics) and write the answer from that list, covering \
every topic. For a single-fact question, reply with the final answer once \
you have it. Every answer follows all the rules above."""


_SYSTEM_PROMPT = _ANSWER_RULES + _AGENT_INSTRUCTIONS

_FORCE_PLAN = {"type": "function", "function": {"name": PLAN_TOOL_NAME}}
_FORCE_TOPICS = {"type": "function", "function": {"name": TOPICS_TOOL_NAME}}


@dataclass
class AgentRunResult:
    answer: str
    sources: List[RetrievedChunk]
    tool_calls: List[ToolCallRecord]
    iterations: int
    plan: Optional[Dict[str, Any]] = None  # plan_research arguments from the first turn
    topics: Optional[List[Dict[str, Any]]] = None  # list_topics inventory, broad questions only
    # Passages collapsed as repeats of an earlier source from another filing:
    # (chunk, [Source N numbers it repeats]). Not numbered, not in the prompt.
    repeated_sources: List[Tuple[RetrievedChunk, List[int]]] = field(default_factory=list)

    @property
    def calculations(self) -> List[Dict[str, Any]]:
        """Successful calculator results, in the order they were computed."""
        return [r.result for r in self.tool_calls if r.is_calculation and r.result is not None]


def _assistant_message(message) -> Dict[str, Any]:
    """Serialize an assistant turn (with its tool calls) back into the history."""
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ],
    }


def _catalog_block(filings: Sequence[FilingPeriod]) -> str:
    lines = []
    for f in filings:
        span = (
            f"{f.period_start:%Y-%m-%d} to {f.period_end:%Y-%m-%d}"
            if f.period_start and f.period_end else "period unknown"
        )
        lines.append(f"{f.company} {f.year} {f.quarter}: {span}")
    return "FILING CATALOG (label: fiscal quarter covered)\n" + "\n".join(lines)


def run_agent(
    query: str,
    search_fn: SearchFn,
    config: AgentConfig,
    max_iterations: Optional[int] = None,
    filings: Optional[Sequence[FilingPeriod]] = None,
) -> AgentRunResult:
    """
    Answer `query` with a tool-calling loop.

    Args:
        query:          The user's question.
        search_fn:      Retrieval callable: search_fn(query, company=, year=,
                        quarter=) -> List[RetrievedChunk]. Wrap the caller's
                        own configured retrieval pipeline.
        config:         Agent model settings.
        max_iterations: Override config.max_iterations (tool-calling turns
                        before a final answer is forced).
        filings:        Filing catalog (see retriever.list_filing_periods):
                        which filings exist and the fiscal quarter each one
                        covers, so the planner can map a question's periods
                        to filings instead of guessing.

    Returns:
        AgentRunResult with the final answer, every source the agent
        retrieved (index i is [Source i+1] in the answer), and the trace of
        tool calls it made.
    """
    max_iterations = max_iterations or config.max_iterations
    executor = AgentToolExecutor(search_fn)
    client = _get_client(config.api_key)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{_catalog_block(filings)}\n\nQuestion: {query}" if filings else f"Question: {query}"
        )},
    ]

    def complete(tool_choice: Any, model: str, tools: List[Dict[str, Any]] = AGENT_TOOLS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        return response.choices[0].message

    def run_tool_calls(message) -> None:
        messages.append(_assistant_message(message))
        for tool_call in message.tool_calls:
            content = executor.execute(tool_call.function.name, tool_call.function.arguments)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": content})

    def needs_topics() -> bool:
        # Overview / multi-part / multi-period questions, per the plan
        plan = executor.plan or {}
        return bool(plan.get("compare_across_periods")) or len(plan.get("sub_topics") or []) > 1

    def finish(turns: int) -> AgentRunResult:
        """List topics if required, then force the final answer."""
        if needs_topics():
            # The inventory decides what the answer covers, so like the plan
            # it runs on the stronger planner model. One retry if it left
            # sources unaccounted for (the tool error names them).
            for _ in range(2):
                run_tool_calls(complete(_FORCE_TOPICS, config.planner_model))
                turns += 1
                if executor.topics is not None:
                    break
        message = complete("none", config.model)
        return _result(message.content, executor, turns + 1)

    for iteration in range(1, max_iterations + 1):
        # Turn 1 is forced to call plan_research, which shapes everything
        # after it, so it runs on the stronger planner model; the searches,
        # calculations and answer use the regular agent model.
        if iteration == 1:
            message = complete(_FORCE_PLAN, config.planner_model)
        else:
            message = complete("auto", config.model, SEARCH_TURN_TOOLS)
        if not message.tool_calls:
            if needs_topics():
                # Searching is done ("READY", or a draft answer, which is
                # discarded): list topics, then write the answer from them.
                return finish(iteration)
            return _result(message.content, executor, iteration)

        run_tool_calls(message)

    logger.warning(
        "Agent hit max_iterations=%d for %r — forcing a final answer from %d sources",
        max_iterations, query, len(executor.sources.chunks),
    )
    return finish(max_iterations)


def _result(content: Optional[str], executor: AgentToolExecutor, iterations: int) -> AgentRunResult:
    result = AgentRunResult(
        answer=(content or "").strip(),
        sources=executor.sources.chunks,
        tool_calls=list(executor.records),
        iterations=iterations,
        plan=executor.plan,
        topics=executor.topics,
        repeated_sources=executor.sources.repeats,
    )
    logger.info(
        "Agent finished in %d LLM turn(s): %d tool call(s), %d source(s), %d calculation(s)",
        iterations, len(result.tool_calls), len(result.sources), len(result.calculations),
    )
    return result
