from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class CopilotStateDB:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.connection:
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA busy_timeout=5000")
            self.connection.execute("""CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY, created_at REAL NOT NULL, event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL)""")
            self.connection.execute("""CREATE TABLE IF NOT EXISTS pending_operations(
                id TEXT PRIMARY KEY, operation_type TEXT NOT NULL, payload_json TEXT NOT NULL,
                expires_at REAL NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL)""")
            self.connection.execute("""CREATE TABLE IF NOT EXISTS runtime_state(
                state_key TEXT PRIMARY KEY, payload_json TEXT NOT NULL,
                updated_at REAL NOT NULL)""")

    def event(self, event_type: str, payload: dict[str, Any]) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT INTO events(created_at,event_type,payload_json) VALUES(?,?,?)",
                (time.time(), event_type, json.dumps(payload, ensure_ascii=False)))

    def set_pending(self, operation_type: str, payload: dict[str, Any], ttl_s: int = 300) -> str:
        operation_id = uuid.uuid4().hex
        now = time.time()
        with self.lock, self.connection:
            self.connection.execute(
                "UPDATE pending_operations SET status='SUPERSEDED' WHERE status='WAITING'")
            self.connection.execute(
                "INSERT INTO pending_operations VALUES(?,?,?,?,?,?)",
                (operation_id, operation_type, json.dumps(payload, ensure_ascii=False),
                 now + ttl_s, "WAITING", now))
        return operation_id

    def pending(self) -> dict[str, Any] | None:
        now = time.time()
        with self.lock, self.connection:
            self.connection.execute(
                "UPDATE pending_operations SET status='EXPIRED' "
                "WHERE status='WAITING' AND expires_at<?", (now,))
            row = self.connection.execute(
                "SELECT * FROM pending_operations WHERE status='WAITING' "
                "ORDER BY created_at DESC LIMIT 1").fetchone()
        if row is None:
            return None
        return {"id": row["id"], "operation_type": row["operation_type"],
                "payload": json.loads(row["payload_json"]),
                "expires_at": row["expires_at"]}

    def resolve_pending(self, status: str) -> dict[str, Any] | None:
        pending = self.pending()
        if pending is None:
            return None
        with self.lock, self.connection:
            self.connection.execute(
                "UPDATE pending_operations SET status=? WHERE id=?", (status, pending["id"]))
        return pending

    def save_runtime_state(self, state_key: str, payload: dict[str, Any]) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT INTO runtime_state(state_key,payload_json,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(state_key) DO UPDATE SET payload_json=excluded.payload_json, "
                "updated_at=excluded.updated_at",
                (state_key, json.dumps(payload, ensure_ascii=False), time.time()))

    def load_runtime_state(self, state_key: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT payload_json FROM runtime_state WHERE state_key=?", (state_key,)
            ).fetchone()
        return json.loads(row["payload_json"]) if row is not None else None

    def close(self) -> None:
        self.connection.close()
