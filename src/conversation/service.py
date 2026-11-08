"""
ConversationManager — runs one turn of a conversation around any single-turn
answerer (the agent), which stays unaware of history:

    message
      -> no history?            answer it as-is (no extra LLM call)
      -> contextualize()         kind + standalone question  [cheap model]
           chit_chat             -> direct reply, no retrieval
           about_previous_answer -> rework the last answer, no retrieval
           new_question / follow_up
                                 -> answer_fn(standalone question)  [agent]
      -> store the turn, update entity state, fold evicted turns into the summary
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.config import ConversationConfig
from src.conversation import contextualizer
from src.conversation.memory import fold_into_summary, render_history, update_state
from src.conversation.models import (
    ABOUT_PREVIOUS_ANSWER, CHIT_CHAT, NEW_QUESTION, Conversation, Turn,
)
from src.conversation.store import ConversationStore

logger = logging.getLogger(__name__)


@dataclass
class AnswerOutcome:
    """What a single-turn answerer returns to the manager."""
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)   # [{"source_file", "pages"}]
    companies: List[str] = field(default_factory=list)
    periods: List[str] = field(default_factory=list)
    payload: Any = None            # the answerer's own result (e.g. AgentRunResult), passed through


@dataclass
class TurnOutcome:
    conversation_id: str
    kind: str
    standalone_question: str
    answer: str
    turn_index: int
    payload: Any = None            # AnswerOutcome.payload when the answerer ran, else None


class ConversationNotFound(KeyError):
    pass


class ConversationManager:
    def __init__(self, store: ConversationStore, config: ConversationConfig) -> None:
        self.store = store
        self.config = config

    def get(self, conversation_id: str) -> Optional[Conversation]:
        return self.store.get(conversation_id)

    def delete(self, conversation_id: str) -> bool:
        return self.store.delete(conversation_id)

    def handle(
        self,
        message: str,
        answer_fn: Callable[[str], AnswerOutcome],
        conversation_id: Optional[str] = None,
    ) -> TurnOutcome:
        if conversation_id:
            conversation = self.store.get(conversation_id)
            if conversation is None:
                raise ConversationNotFound(conversation_id)
        else:
            conversation = Conversation(id=uuid.uuid4().hex)

        cfg = self.config
        if conversation.turns:
            history = render_history(conversation, cfg.window_turns, cfg.max_answer_chars)
            ctx = contextualizer.contextualize(message, history, cfg)
        else:
            ctx = contextualizer.Contextualized(NEW_QUESTION, message)

        outcome: Optional[AnswerOutcome] = None
        if ctx.kind == CHIT_CHAT and ctx.direct_reply:
            answer = ctx.direct_reply
        elif ctx.kind == ABOUT_PREVIOUS_ANSWER and conversation.last_turn:
            answer = contextualizer.answer_about_previous(
                message, conversation.last_turn, cfg.light_model, cfg.api_key, cfg.max_tokens,
            )
        else:
            outcome = answer_fn(ctx.standalone_question)
            answer = outcome.answer

        turn = Turn(
            user_message=message,
            standalone_question=ctx.standalone_question,
            kind=ctx.kind,
            answer=answer,
            sources=outcome.sources if outcome else [],
            companies=outcome.companies if outcome else [],
            periods=outcome.periods if outcome else [],
        )
        conversation.turns.append(turn)
        update_state(conversation, turn)
        fold_into_summary(
            conversation, cfg.window_turns,
            lambda summary, turns: contextualizer.summarize_turns(summary, turns, cfg.light_model, cfg.api_key),
        )
        self.store.save(conversation)

        return TurnOutcome(
            conversation_id=conversation.id,
            kind=ctx.kind,
            standalone_question=ctx.standalone_question,
            answer=answer,
            turn_index=len(conversation.turns) - 1,
            payload=outcome.payload if outcome else None,
        )
