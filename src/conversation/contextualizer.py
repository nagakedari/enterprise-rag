"""
LLM steps that make a message answerable in the context of a conversation,
built as LangChain (LCEL) prompt chains:

  contextualize()  — classify -> branch -> rewrite:

      {message, history}
        | classify       Jev (TypeSafe's own API, /v1/systemone): a
                         typed choice over the turn kinds with probabilities —
                         cheap and fast. with_fallbacks -> a gpt-4o
                         classification prompt if Jev is unavailable
                         (no TYPESAFE_API_KEY, HTTP error, unknown label).
        | branch
            follow_up             -> rewrite chain (CONVERSATION_MODEL, gpt-4o):
                                     prompt does query rewriting ONLY, plus a
                                     guard that retries once if an explicit
                                     quarter label was dropped
            chit_chat             -> short reply (light model; Jev returns labels, not text)
            new_question /
            about_previous_answer -> message passed through unchanged

  answer_about_previous()  rephrase / shorten / explain the last answer,
                           using only that answer (no retrieval)  [light model]
  summarize_turns()        fold old turns into the running summary [light model]

None of them produce financial facts of their own. gpt-4o rewrites because
gpt-4o-mini computed calendar dates for relative periods in testing.
"""
from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.config import ConversationConfig
from src.conversation.models import (
    ABOUT_PREVIOUS_ANSWER, CHIT_CHAT, FOLLOW_UP, NEW_QUESTION, TURN_KINDS, Turn,
)
from src.generation.generator import _get_client

logger = logging.getLogger(__name__)


@dataclass
class Contextualized:
    kind: str
    standalone_question: str
    direct_reply: str = ""
    classifier: str = ""                 # "jev" | "openai" | "" (first turn, not classified)
    confidence: Optional[float] = None   # Jev's confidence over the choice distribution


# ── 1. Classification ─────────────────────────────────────────────────────────

# Criteria for Jev's choice question (and the fallback prompt): label -> description
_KIND_CRITERIA: Dict[str, str] = {
    NEW_QUESTION: (
        "The new message is fully understandable on its own: it states what it asks "
        "about (company, metric, period or subject) without relying on earlier turns."
    ),
    FOLLOW_UP: (
        "The new message leaves something implicit that earlier turns supply and needs "
        "new data: pronouns like it/that, 'what about X?', 'and Q1?', 'the quarter before', "
        "'compare that with…', 'why?' — even when it changes the company or metric."
    ),
    ABOUT_PREVIOUS_ANSWER: (
        "The new message asks to rephrase, shorten, expand, format, translate or explain "
        "the assistant's previous answer, without needing any new data."
    ),
    CHIT_CHAT: "The new message is a greeting, thanks or small talk.",
}
_KIND_INSTRUCTIONS = (
    "How does the NEW MESSAGE relate to the conversation about companies' SEC 10-Q "
    "filings shown before it?"
)


class JevUnavailable(RuntimeError):
    pass


def _state_text(inputs: Dict[str, Any]) -> str:
    return f"{inputs['history']}\n\nNEW MESSAGE: {inputs['message']}"


def _jev_classify(inputs: Dict[str, Any], cfg: ConversationConfig) -> Dict[str, Any]:
    """One typed choice question to Jev; raises (-> fallback) on any problem."""
    if not cfg.jev_api_key:
        raise JevUnavailable("TYPESAFE_API_KEY is not set")
    try:
        response = httpx.post(
            cfg.jev_endpoint,
            headers={"Authorization": f"Bearer {cfg.jev_api_key}"},
            json={
                "model": cfg.jev_model,
                "state": _state_text(inputs),
                "questions": {"kind": {
                    "type": "choice", "instructions": _KIND_INSTRUCTIONS, "criteria": _KIND_CRITERIA,
                }},
            },
            timeout=cfg.jev_timeout,
        )
        response.raise_for_status()
        answer = response.json()["answers"]["kind"]
        kind = answer.get("choice")
        if kind not in TURN_KINDS:
            raise JevUnavailable(f"unexpected Jev choice {kind!r}")
    except Exception as exc:
        logger.warning("Jev classification failed (%s) — falling back to %s", exc, cfg.model)
        raise
    return {"kind": kind, "confidence": answer.get("confidence"), "classifier": "jev"}


_FALLBACK_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Classify the NEW MESSAGE in a conversation about companies' SEC 10-Q filings. "
               "Kinds:\n" + "\n".join(f"- {k}: {v}" for k, v in _KIND_CRITERIA.items())),
    ("human", "{history}\n\nNEW MESSAGE: {message}"),
])
_KIND_SCHEMA = {
    "title": "turn_kind",
    "description": "The kind of the new message.",
    "type": "object",
    "properties": {"kind": {"type": "string", "enum": TURN_KINDS}},
    "required": ["kind"],
}


def _classify_chain(cfg: ConversationConfig) -> Runnable:
    llm_classify = (
        _FALLBACK_CLASSIFY_PROMPT
        | _llm(cfg.model, cfg.api_key).with_structured_output(_KIND_SCHEMA)
        | RunnableLambda(lambda out: {
            "kind": out.get("kind") if out.get("kind") in TURN_KINDS else FOLLOW_UP,
            "confidence": None, "classifier": "openai",
        })
    )
    if cfg.classifier != "jev":
        return llm_classify
    jev = RunnableLambda(lambda inputs: _jev_classify(inputs, cfg))
    return jev.with_fallbacks([llm_classify])


# ── 2. Rewriting (query rewriting only) ───────────────────────────────────────

_CONTEXTUALIZE_PROMPT = """You rewrite a follow-up message from a conversation about \
companies' SEC 10-Q filings into a standalone question for a research assistant. \
You do NOT answer it. Reply with ONLY the rewritten question.

Rules:
- Name the company, metric and period explicitly, resolved only from the \
conversation shown. Never add facts, figures, dates or assumptions of your own, \
and never narrow or change what is being asked.
- Ask exactly what the user asked — keep words like compare, why, how did it \
change, trend.
- Carry over only what the message leaves implicit: for "what about Intel?" after \
a question on Apple's Mac revenue for a quarter, keep the metric and period but \
replace the company.
- PERIODS: (a) an EXPLICIT period the user names — a filing label like "Q1" or \
"Q4 2023", a fiscal year, or a date — is kept exactly as the user wrote it; only \
fill in a missing part from the conversation (e.g. the year: "q1" after "Q3 2023" \
-> "Q1 2023"). Never turn an explicit period into a relative phrase or a date. \
(b) Only when the user uses a RELATIVE expression ("the quarter before", \
"previous quarter", "next quarter", "same quarter last year") describe it relative \
to the period it refers to (e.g. "the quarter immediately before Q3 2023") — never \
compute a new date. (c) A message with no period inherits the period of the turn \
it follows up.{correction}"""

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _CONTEXTUALIZE_PROMPT),
    ("human", "{history}\n\nNEW MESSAGE: {message}"),
])

# Explicit quarter labels in a user message ("q1", "Q4 2023"). Filing labels
# differ from calendar quarters and by company, so a label the user names must
# reach the agent as a label — never re-expressed as a date or "the quarter
# before X" (which once turned "q1" into Q2).
_QUARTER_LABEL_RE = re.compile(r"\bq([1-4])\b", re.IGNORECASE)


def _missing_labels(message: str, standalone: str) -> List[str]:
    asked = {f"Q{n}" for n in _QUARTER_LABEL_RE.findall(message)}
    kept = {f"Q{n}" for n in _QUARTER_LABEL_RE.findall(standalone)}
    return sorted(asked - kept)


def _rewrite_chain(cfg: ConversationConfig) -> Runnable:
    rewrite = _REWRITE_PROMPT | _llm(cfg.model, cfg.api_key) | StrOutputParser()

    def with_label_guard(inputs: Dict[str, Any]) -> str:
        standalone = rewrite.invoke({**inputs, "correction": ""}).strip()
        missing = _missing_labels(inputs["message"], standalone)
        if missing:
            # One corrective retry, only when an explicit label was dropped
            logger.info("Rewrite dropped quarter label(s) %s for %r — retrying", missing, inputs["message"])
            labels = ", ".join(missing)
            standalone = rewrite.invoke({**inputs, "correction": (
                f"\n\nCORRECTION: the user explicitly named {labels}. The rewritten question "
                f"must contain {labels} as written (add the year from the conversation if "
                "missing) — not a date and not 'the quarter before/after'.")}).strip()
        return standalone or inputs["message"]

    return RunnableLambda(with_label_guard)


# ── 3. Chit-chat reply ────────────────────────────────────────────────────────

_CHIT_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant for companies' SEC 10-Q filings. Reply to "
               "the user's small talk in one or two friendly sentences; offer help with "
               "questions about the filings. No financial facts."),
    ("human", "{message}"),
])


# ── Full chain ────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=4)
def _llm(model: str, api_key: str) -> ChatOpenAI:
    return ChatOpenAI(model=model, api_key=api_key, temperature=0)


def _contextualize_chain(cfg: ConversationConfig) -> Runnable:
    is_kind = lambda kind: (lambda x: x["classification"]["kind"] == kind)
    return RunnablePassthrough.assign(classification=_classify_chain(cfg)) | RunnableBranch(
        (is_kind(FOLLOW_UP), RunnablePassthrough.assign(standalone=_rewrite_chain(cfg))),
        (is_kind(CHIT_CHAT), RunnablePassthrough.assign(
            standalone=lambda x: x["message"],
            direct_reply=_CHIT_CHAT_PROMPT | _llm(cfg.light_model, cfg.api_key) | StrOutputParser(),
        )),
        # new_question / about_previous_answer: the message itself, never rewritten
        RunnablePassthrough.assign(standalone=lambda x: x["message"]),
    )


def contextualize(message: str, history: str, cfg: ConversationConfig) -> Contextualized:
    out = _contextualize_chain(cfg).invoke({"message": message, "history": history})
    c = out["classification"]
    result = Contextualized(
        kind=c["kind"], standalone_question=out["standalone"], direct_reply=(out.get("direct_reply") or "").strip(),
        classifier=c["classifier"], confidence=c.get("confidence"),
    )
    logger.info("Contextualized %r -> %s (%s, confidence=%s): %r", message, result.kind,
                result.classifier, result.confidence, result.standalone_question)
    return result


def _complete(model: str, api_key: str, messages, **kw):
    return _get_client(api_key).chat.completions.create(model=model, messages=messages, temperature=0, **kw)


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
