from __future__ import annotations

import io
import json
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class StreamingPreview:
    """Best-effort partial transcription worker for speculative previews.

    It never blocks VAD capture and drops a preview while the previous one is
    still being transcribed. Final STT remains authoritative.
    """

    def __init__(
        self,
        transcriber: Any,
        *,
        sample_rate: int = 16_000,
        on_text: Callable[[str, int], None] | None = None,
    ) -> None:
        self.transcriber = transcriber
        self.sample_rate = sample_rate
        self.on_text = on_text or self._print
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._busy = False
        self._lock = threading.Lock()

    def __call__(self, pcm: bytes, duration_ms: int) -> None:
        with self._lock:
            if self._busy:
                return
            self._busy = True
        self._executor.submit(self._transcribe, pcm, duration_ms)

    def _transcribe(self, pcm: bytes, duration_ms: int) -> None:
        try:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(self.sample_rate)
                output.writeframes(pcm)
            text = self.transcriber.transcribe(buffer.getvalue()).strip()
            if text:
                self.on_text(text, duration_ms)
        except Exception:
            # Partial previews are optional; final STT must remain unaffected.
            pass
        finally:
            with self._lock:
                self._busy = False

    @staticmethod
    def _print(text: str, duration_ms: int) -> None:
        print(
            json.dumps(
                {"event": "PARTIAL_TRANSCRIPT", "text": text, "duration_ms": duration_ms},
                ensure_ascii=False,
            ),
            flush=True,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
