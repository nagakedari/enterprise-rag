"""Data model for multi-turn conversations (see MULTI_TURN_STRATEGIES.md)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# How a new message relates to the conversation (decided by the contextualizer)
NEW_QUESTION = "new_question"                    # self-contained, unrelated to earlier turns
FOLLOW_UP = "follow_up"                          # needs earlier turns to be understood
ABOUT_PREVIOUS_ANSWER = "about_previous_answer"  # rephrase/shorten/explain the last answer — no new data
CHIT_CHAT = "chit_chat"                          # greeting, thanks, small talk
TURN_KINDS = [NEW_QUESTION, FOLLOW_UP, ABOUT_PREVIOUS_ANSWER, CHIT_CHAT]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Turn:
    """One user message and the assistant's answer to it."""
    user_message: str
    standalone_question: str          # what was actually answered (follow-ups resolved)
    kind: str
    answer: str
    # Compact record of what backed the answer: [{"source_file", "pages"}, ...]
    sources: List[Dict[str, Any]] = field(default_factory=list)
    # Companies / filing labels the agent's plan used — feeds conversation state
    companies: List[str] = field(default_factory=list)
    periods: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utcnow)


@dataclass
class Conversation:
    id: str
    turns: List[Turn] = field(default_factory=list)
    # Running summary of turns that have left the verbatim window
    summary: str = ""
    summarized_turns: int = 0          # how many leading turns the summary covers
    # Active entities carried across turns (latest first): {"companies": [...], "periods": [...]}
    state: Dict[str, List[str]] = field(default_factory=lambda: {"companies": [], "periods": []})
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)

    @property
    def last_turn(self) -> Optional[Turn]:
        return self.turns[-1] if self.turns else None
