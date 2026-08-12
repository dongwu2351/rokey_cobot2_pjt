"""Small persistent memory for conversation turns (text only, no audio)."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path


class ConversationMemory:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                route TEXT
            )"""
        )
        self.connection.commit()

    def add(self, role: str, content: str, route: str | None = None) -> None:
        with self.lock:
            self.connection.execute(
                "INSERT INTO conversation_turns(created_at, role, content, route) VALUES (?, ?, ?, ?)",
                (time.time(), role, content.strip(), route),
            )
            self.connection.commit()

    def recent(self, limit: int = 12) -> list[dict[str, str]]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT role, content FROM conversation_turns ORDER BY id DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [
            {"role": str(role), "content": str(content)}
            for role, content in reversed(rows)
        ]

    def close(self) -> None:
        with self.lock:
            self.connection.close()
