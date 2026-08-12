from __future__ import annotations

import difflib
import json
import sqlite3
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def normalize_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).lower().split())


@dataclass(frozen=True)
class RememberedObject:
    canonical_name: str
    grounding_prompt: str
    aliases: tuple[str, ...]
    attributes: dict[str, str | int | float | bool | None]
    recall_count: int
    last_seen_ms: int


class ObjectMemory:
    """Persistent semantic memory for concepts, never for stale robot coordinates."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS object_concepts (
                canonical_name TEXT PRIMARY KEY,
                grounding_prompt TEXT NOT NULL,
                aliases_json TEXT NOT NULL,
                attributes_json TEXT NOT NULL,
                recall_count INTEGER NOT NULL DEFAULT 0,
                last_seen_ms INTEGER NOT NULL
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "ObjectMemory":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def remember(
        self,
        canonical_name: str,
        *,
        grounding_prompt: str,
        aliases: Iterable[str] = (),
        attributes: dict[str, str | int | float | bool | None] | None = None,
        now_ms: int | None = None,
    ) -> RememberedObject:
        canonical = normalize_name(canonical_name)
        prompt = grounding_prompt.strip()
        if not canonical or not prompt:
            raise ValueError("canonical_name and grounding_prompt are required")
        normalized_aliases = tuple(
            dict.fromkeys(
                value
                for value in (normalize_name(alias) for alias in aliases)
                if value and value != canonical
            )
        )
        timestamp = now_ms if now_ms is not None else round(time.time() * 1_000)
        payload_attributes = attributes or {}
        with self._lock:
            existing = self._connection.execute(
                "SELECT aliases_json FROM object_concepts WHERE canonical_name = ?",
                (canonical,),
            ).fetchone()
            if existing is not None:
                previous_aliases = tuple(json.loads(existing[0]))
                normalized_aliases = tuple(
                    dict.fromkeys((*previous_aliases, *normalized_aliases))
                )
            self._connection.execute(
                """
                INSERT INTO object_concepts (
                    canonical_name, grounding_prompt, aliases_json,
                    attributes_json, recall_count, last_seen_ms
                ) VALUES (?, ?, ?, ?, 0, ?)
                ON CONFLICT(canonical_name) DO UPDATE SET
                    grounding_prompt=excluded.grounding_prompt,
                    aliases_json=excluded.aliases_json,
                    attributes_json=excluded.attributes_json,
                    last_seen_ms=excluded.last_seen_ms
                """,
                (
                    canonical,
                    prompt,
                    json.dumps(normalized_aliases, ensure_ascii=False),
                    json.dumps(payload_attributes, ensure_ascii=False, sort_keys=True),
                    timestamp,
                ),
            )
            self._connection.commit()
        return self.get(canonical)  # type: ignore[return-value]

    def get(self, canonical_name: str) -> RememberedObject | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM object_concepts WHERE canonical_name = ?",
                (normalize_name(canonical_name),),
            ).fetchone()
        return self._from_row(row) if row else None

    def recall(self, query: str, *, minimum_similarity: float = 0.86) -> RememberedObject | None:
        normalized = normalize_name(query)
        if not normalized:
            return None
        with self._lock:
            rows = self._connection.execute("SELECT * FROM object_concepts").fetchall()
            best_row = None
            best_score = 0.0
            for row in rows:
                names = (row["canonical_name"], *json.loads(row["aliases_json"]))
                score = max(
                    1.0 if normalized == name else difflib.SequenceMatcher(
                        None, normalized, name
                    ).ratio()
                    for name in names
                )
                if score > best_score:
                    best_score, best_row = score, row
            if best_row is None or best_score < minimum_similarity:
                return None
            self._connection.execute(
                "UPDATE object_concepts SET recall_count = recall_count + 1 "
                "WHERE canonical_name = ?",
                (best_row["canonical_name"],),
            )
            self._connection.commit()
            refreshed = self._connection.execute(
                "SELECT * FROM object_concepts WHERE canonical_name = ?",
                (best_row["canonical_name"],),
            ).fetchone()
        return self._from_row(refreshed)

    def recent(self, *, limit: int = 20) -> tuple[RememberedObject, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM object_concepts ORDER BY last_seen_ms DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(self._from_row(row) for row in rows)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> RememberedObject:
        return RememberedObject(
            canonical_name=row["canonical_name"],
            grounding_prompt=row["grounding_prompt"],
            aliases=tuple(json.loads(row["aliases_json"])),
            attributes=json.loads(row["attributes_json"]),
            recall_count=row["recall_count"],
            last_seen_ms=row["last_seen_ms"],
        )
