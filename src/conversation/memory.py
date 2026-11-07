"""
Conversation memory: what of the past is shown to the model on a new turn.

Three layers, cheapest first (see MULTI_TURN_STRATEGIES.md):

  1. Entity state  — companies and filing labels the agent's plans used,
                     latest first. Deterministic, no LLM call. Lets "and the
                     quarter before?" resolve to the right company.
  2. Sliding window — the last `window_turns` turns verbatim (answers
                     truncated), so recent references resolve exactly.
  3. Running summary — turns that leave the window are folded into a short
                     summary by a cheap model, one call per evicted turn, so
                     long conversations keep their gist at a bounded size.
"""
from __future__ import annotations

import logging
from typing import Callable, List

from src.conversation.models import Conversation, Turn

logger = logging.getLogger(__name__)

_MAX_STATE_ITEMS = 6


def update_state(conversation: Conversation, turn: Turn) -> None:
    """Merge the turn's companies / filing labels into the state, latest first."""
    for key, values in (("companies", turn.companies), ("periods", turn.periods)):
        merged = list(dict.fromkeys(list(values) + conversation.state.get(key, [])))
        conversation.state[key] = merged[:_MAX_STATE_ITEMS]


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def render_history(conversation: Conversation, window_turns: int, max_answer_chars: int) -> str:
    """Summary + recent turns + active entities, as plain text for a prompt."""
    parts: List[str] = []
    if conversation.summary:
        parts.append(f"EARLIER IN THE CONVERSATION (summary):\n{conversation.summary}")
    recent = conversation.turns[-window_turns:] if window_turns > 0 else []
    if recent:
        lines = []
        for t in recent:
            lines.append(f"User: {t.user_message}")
            if t.standalone_question and t.standalone_question != t.user_message:
                lines.append(f"(understood as: {t.standalone_question})")
            lines.append(f"Assistant: {_truncate(t.answer, max_answer_chars)}")
        parts.append("RECENT TURNS (oldest first):\n" + "\n".join(lines))
    companies, periods = conversation.state.get("companies"), conversation.state.get("periods")
    if companies or periods:
        parts.append(
            "ACTIVE CONTEXT (most recent first): "
            f"companies={', '.join(companies) or '-'}; filings={', '.join(periods) or '-'}"
        )
    return "\n\n".join(parts)


def fold_into_summary(
    conversation: Conversation,
    window_turns: int,
    summarize: Callable[[str, List[Turn]], str],
) -> None:
    """
    Fold turns that fell out of the verbatim window into the running summary.
    `summarize(previous_summary, turns) -> new_summary` is the (cheap) LLM call;
    it runs only when there are newly evicted turns.
    """
    evict_upto = max(0, len(conversation.turns) - window_turns)
    if evict_upto <= conversation.summarized_turns:
        return
    evicted = conversation.turns[conversation.summarized_turns:evict_upto]
    try:
        conversation.summary = summarize(conversation.summary, evicted)
        conversation.summarized_turns = evict_upto
    except Exception:
        # Memory is best-effort: keep the old summary and retry on a later turn
        logger.exception("Conversation summary update failed for %s", conversation.id)
