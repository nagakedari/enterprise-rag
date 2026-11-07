"""
Conversation storage behind one small interface, so the backend can change
without touching the API or the memory logic:

    SQLiteConversationStore   default — persistent, single file, no extra
                              infrastructure (storage/conversations.db)
    InMemoryConversationStore tests / throwaway local runs; lost on restart

A Redis or Postgres store only has to implement the same four methods (see
MULTI_TURN_STRATEGIES.md for when to move off SQLite).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Protocol

from src.conversation.models import Conversation, Turn, utcnow


class ConversationStore(Protocol):
    def get(self, conversation_id: str) -> Optional[Conversation]: ...
    def save(self, conversation: Conversation) -> None: ...
    def delete(self, conversation_id: str) -> bool: ...
    def purge_older_than(self, days: int) -> int: ...


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._data: Dict[str, Conversation] = {}
        self._lock = threading.Lock()

    def get(self, conversation_id: str) -> Optional[Conversation]:
        with self._lock:
            return self._data.get(conversation_id)

    def save(self, conversation: Conversation) -> None:
        conversation.updated_at = utcnow()
        with self._lock:
            self._data[conversation.id] = conversation

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            return self._data.pop(conversation_id, None) is not None

    def purge_older_than(self, days: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._lock:
            stale = [k for k, c in self._data.items() if c.updated_at < cutoff]
            for k in stale:
                del self._data[k]
        return len(stale)


class SQLiteConversationStore:
    """
    One row per conversation (summary + state as JSON) and one row per turn.
    A connection per call keeps it safe under FastAPI's thread pool; WAL mode
    lets reads proceed while a turn is being written.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id               TEXT PRIMARY KEY,
                    created_at       TEXT NOT NULL,
                    updated_at       TEXT NOT NULL,
                    summary          TEXT NOT NULL DEFAULT '',
                    summarized_turns INTEGER NOT NULL DEFAULT 0,
                    state            TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS turns (
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    idx             INTEGER NOT NULL,
                    data            TEXT NOT NULL,
                    PRIMARY KEY (conversation_id, idx)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self._path, timeout=10)
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def get(self, conversation_id: str) -> Optional[Conversation]:
        with self._connect() as db:
            row = db.execute(
                "SELECT id, created_at, updated_at, summary, summarized_turns, state "
                "FROM conversations WHERE id = ?", (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            turns = [
                Turn(**json.loads(data))
                for (data,) in db.execute(
                    "SELECT data FROM turns WHERE conversation_id = ? ORDER BY idx", (conversation_id,),
                )
            ]
        return Conversation(
            id=row[0], created_at=row[1], updated_at=row[2], summary=row[3],
            summarized_turns=row[4], state=json.loads(row[5]), turns=turns,
        )

    def save(self, conversation: Conversation) -> None:
        conversation.updated_at = utcnow()
        with self._connect() as db:
            db.execute(
                "INSERT INTO conversations (id, created_at, updated_at, summary, summarized_turns, state) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "updated_at=excluded.updated_at, summary=excluded.summary, "
                "summarized_turns=excluded.summarized_turns, state=excluded.state",
                (conversation.id, conversation.created_at, conversation.updated_at, conversation.summary,
                 conversation.summarized_turns, json.dumps(conversation.state)),
            )
            # Turns are append-only: insert the ones not stored yet
            stored = db.execute(
                "SELECT COUNT(*) FROM turns WHERE conversation_id = ?", (conversation.id,),
            ).fetchone()[0]
            db.executemany(
                "INSERT INTO turns (conversation_id, idx, data) VALUES (?, ?, ?)",
                [(conversation.id, i, json.dumps(asdict(t))) for i, t in enumerate(conversation.turns) if i >= stored],
            )

    def delete(self, conversation_id: str) -> bool:
        with self._connect() as db:
            return db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,)).rowcount > 0

    def purge_older_than(self, days: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._connect() as db:
            return db.execute("DELETE FROM conversations WHERE updated_at < ?", (cutoff,)).rowcount


def make_store(backend: str, path: Path) -> ConversationStore:
    if backend == "memory":
        return InMemoryConversationStore()
    if backend == "sqlite":
        return SQLiteConversationStore(path)
    raise ValueError(f"Unknown CONVERSATION_STORE {backend!r} (use 'sqlite' or 'memory')")
