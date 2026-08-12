from __future__ import annotations

import json
import re
import threading
from typing import Any, Callable


class SpeculativeIntentEngine:
    """Promote only stable partial transcripts to speculative intent events."""

    def __init__(
        self,
        router: Any,
        *,
        on_stable: Callable[[str, Any], None] | None = None,
        repeats_required: int = 2,
        min_chars: int = 4,
    ) -> None:
        self.router = router
        self.on_stable = on_stable or self._print
        self.repeats_required = max(2, repeats_required)
        self.min_chars = max(2, min_chars)
        self._last_normalized = ""
        self._repeat_count = 0
        self._last_emitted = ""
        self._lock = threading.Lock()

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def __call__(self, text: str, duration_ms: int = 0) -> None:
        normalized = self._normalize(text)
        if len(normalized) < self.min_chars:
            return
        with self._lock:
            if normalized == self._last_normalized:
                self._repeat_count += 1
            else:
                self._last_normalized = normalized
                self._repeat_count = 1
            if self._repeat_count < self.repeats_required:
                return
            if normalized == self._last_emitted:
                return
            self._last_emitted = normalized
        try:
            decision = self.router.route(text)
            self.on_stable(text, decision)
        except Exception:
            # Speculation must never interrupt final STT or execution.
            return

    @staticmethod
    def _print(text: str, decision: Any) -> None:
        route = getattr(decision, "route", None)
        print(
            json.dumps(
                {
                    "event": "INTENT_STABLE",
                    "text": text,
                    "route": getattr(route, "value", route),
                    "robot_action_allowed": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
