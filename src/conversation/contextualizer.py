"""
LLM steps that make a message answerable in the context of a conversation.

  contextualize()          follow-up -> standalone question + turn kind
                           [CONVERSATION_MODEL, default gpt-4o: in a live test
                           gpt-4o-mini computed calendar dates for relative
                           periods ("same quarter last year" -> a date outside
                           the corpus); gpt-4o kept them relative]
  answer_about_previous()  rephrase / shorten / explain the last answer,
                           using only that answer (no retrieval)  [light model]
  summarize_turns()        fold old turns into the running summary [light model]
None of them produce financial facts of their own.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import List

from src.conversation.models import (
    ABOUT_PREVIOUS_ANSWER, CHIT_CHAT, FOLLOW_UP, NEW_QUESTION, TURN_KINDS, Turn,
)
from src.generation.generator import _get_client

logger = logging.getLogger(__name__)

_CONTEXTUALIZE_TOOL = {
    "type": "function",
    "function": {
        "name": "contextualize",
        "description": "Classify the new message and rewrite it as a standalone question.",
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": TURN_KINDS,
                    "description": (
                        f"{NEW_QUESTION}: fully understandable on its own — it states what "
                        "it asks about. "
                        f"{FOLLOW_UP}: leaves something implicit that earlier turns supply "
                        "(pronouns like it/that, 'what about X?', 'and the quarter before?', "
                        "'compare that with…', 'why?'), even when it changes the company or "
                        "metric; needs new data. "
                        f"{ABOUT_PREVIOUS_ANSWER}: asks to rephrase, shorten, expand, format, "
                        "translate or explain the previous answer WITHOUT needing new data. "
                        f"{CHIT_CHAT}: greeting, thanks or small talk."
                    ),
                },
                "standalone_question": {
                    "type": "string",
                    "description": (
                        "The message rewritten so it can be understood with no history: name "
                        "the company, metric and period explicitly, resolved from the "
                        "conversation. Ask exactly what the user asked — keep words like "
                        "compare, why, how did it change, trend. PERIODS: (a) an EXPLICIT "
                        "period the user names — a filing label like 'Q1' or 'Q4 2023', a "
                        "fiscal year, or a date — is kept exactly as the user wrote it; only "
                        "fill in a missing part from the conversation (e.g. the year: 'q1' "
                        "after 'Q3 2023' -> 'Q1 2023'). Never turn an explicit period into a "
                        "relative phrase or a date. (b) Only when the user uses a RELATIVE "
                        "expression ('the quarter before', 'previous quarter', 'next quarter', "
                        "'same quarter last year') describe it relative to the period it "
                        "refers to (e.g. 'the quarter immediately before Q3 2023') — never "
                        "compute a new date. (c) A message with no period inherits the period "
                        f"of the turn it follows up. For {NEW_QUESTION}, the message unchanged."
                    ),
                },
                "direct_reply": {
                    "type": "string",
                    "description": f"Only for {CHIT_CHAT}: a brief reply.",
                },
            },
            "required": ["kind", "standalone_question"],
        },
    },
}

_CONTEXTUALIZE_PROMPT = """You prepare messages in a conversation about companies' SEC \
10-Q filings for a research assistant. You do NOT answer questions. Given the \
conversation so far and a new message, call contextualize: classify the message \
and rewrite it as a standalone question that names the company, metric and \
period explicitly. Resolve references only from the conversation shown; never \
add facts, figures, dates or assumptions of your own, and never narrow or change \
what is being asked. Carry over from earlier turns only what the message leaves \
implicit: for "what about Intel?" after a question on Apple's Mac revenue for a \
quarter, keep the metric and period but replace the company."""


@dataclass
class Contextualized:
    kind: str
    standalone_question: str
    direct_reply: str = ""


def _complete(model: str, api_key: str, messages, **kw):
    return _get_client(api_key).chat.completions.create(model=model, messages=messages, temperature=0, **kw)


# Explicit quarter labels in a user message ("q1", "Q4 2023"). Filing labels
# differ from calendar quarters and by company, so a label the user names must
# reach the agent as a label — never re-expressed as a date or "the quarter
# before X" (which once turned "q1" into Q2).
_QUARTER_LABEL_RE = re.compile(r"\bq([1-4])\b", re.IGNORECASE)


def _missing_labels(message: str, standalone: str) -> List[str]:
    asked = {f"Q{n}" for n in _QUARTER_LABEL_RE.findall(message)}
    kept = {f"Q{n}" for n in _QUARTER_LABEL_RE.findall(standalone)}
    return sorted(asked - kept)


def _contextualize_once(message: str, history: str, model: str, api_key: str, note: str = ""):
    response = _complete(
        model, api_key,
        [{"role": "system", "content": _CONTEXTUALIZE_PROMPT + note},
         {"role": "user", "content": f"{history}\n\nNEW MESSAGE: {message}"}],
        tools=[_CONTEXTUALIZE_TOOL],
        tool_choice={"type": "function", "function": {"name": "contextualize"}},
    )
    return json.loads(response.choices[0].message.tool_calls[0].function.arguments or "{}")


def contextualize(message: str, history: str, model: str, api_key: str) -> Contextualized:
    args = _contextualize_once(message, history, model, api_key)
    missing = _missing_labels(message, args.get("standalone_question") or "")
    if missing and args.get("kind") == FOLLOW_UP:
        # One corrective retry, only when an explicit label was dropped
        logger.info("Rewrite dropped quarter label(s) %s for %r — retrying", missing, message)
        args = _contextualize_once(message, history, model, api_key, note=(
            f"\n\nCORRECTION: the user explicitly named {', '.join(missing)}. The standalone "
            f"question must contain {', '.join(missing)} as written (add the year from the "
            "conversation if missing) — not a date and not 'the quarter before/after'."))
    kind = args.get("kind") if args.get("kind") in TURN_KINDS else FOLLOW_UP
    standalone = (args.get("standalone_question") or "").strip() or message
    if kind == NEW_QUESTION:
        # Self-contained already: a rewrite can only drift from what was asked
        standalone = message
    result = Contextualized(kind, standalone, (args.get("direct_reply") or "").strip())
    logger.info("Contextualized %r -> %s: %r", message, result.kind, result.standalone_question)
    return result


def answer_about_previous(message: str, previous: Turn, model: str, api_key: str, max_tokens: int) -> str:
    response = _complete(
        model, api_key,
        [{"role": "system", "content": (
            "You rework a previous answer about SEC 10-Q filings as the user asks "
            "(shorter, simpler, a table, an explanation…). Use ONLY the previous "
            "answer: keep its figures and [Source N] citations exactly, add no new "
            "facts. If the request needs information the previous answer does not "
            "contain, say that a new question is needed to look it up.")},
         {"role": "user", "content": (
            f"PREVIOUS QUESTION: {previous.standalone_question}\n\n"
            f"PREVIOUS ANSWER:\n{previous.answer}\n\nREQUEST: {message}")}],
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def summarize_turns(previous_summary: str, turns: List[Turn], model: str, api_key: str) -> str:
    transcript = "\n".join(
        f"User: {t.standalone_question}\nAssistant: {t.answer}" for t in turns
    )
    response = _complete(
        model, api_key,
        [{"role": "system", "content": (
            "Update a running summary of a conversation about SEC 10-Q filings. "
            "Keep it under 120 words: which companies, metrics and periods were "
            "discussed and the key figures concluded (exactly as stated). No new facts.")},
         {"role": "user", "content": (
            f"CURRENT SUMMARY:\n{previous_summary or '(none)'}\n\nTURNS TO ADD:\n{transcript}")}],
        max_tokens=250,
    )
    return (response.choices[0].message.content or "").strip()
