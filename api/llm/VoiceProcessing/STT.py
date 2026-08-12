from __future__ import annotations

import os
import io
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from openai import OpenAI


ENV_PATH = Path(__file__).with_name(".env")
DEFAULT_KEYWORDS = (
    "ROKEY", "조립", "작업", "단계", "부품", "컨베이어", "모터", "풀리",
    "타이밍벨트", "다공판", "브래킷", "RealSense", "AR 창", "에이알 창",
    "참고 사진", "매뉴얼",
    "해머", "망치", "드라이버", "렌치", "플라이어", "드릴",
)


@dataclass(frozen=True)
class STTConfig:
    model: str = "gpt-transcribe"
    language: str = "ko"
    timeout_seconds: float = 8.0
    prompt: str = (
        "한국어 자유 대화 및 조립 작업 코파일럿 발화입니다. 질문, 정정, 후속 문장과 "
        "조립 단계·부품·공구 이름을 문맥 그대로 정확하게 전사하세요."
    )
    keywords: tuple[str, ...] = DEFAULT_KEYWORDS
    backend: str = "cloud"
    local_model: str = "base"
    local_device: str = "cpu"
    local_compute_type: str = "int8"
    local_beam_size: int = 1


class STT:
    """OpenAI transcription client that consumes completed in-memory WAV data."""

    def __init__(
        self,
        openai_api_key: str | None = None,
        *,
        config: STTConfig | None = None,
        client: Any | None = None,
    ) -> None:
        load_dotenv(ENV_PATH)
        self.config = config or STTConfig(
            model=os.getenv("STT_MODEL", "gpt-transcribe"),
            backend=os.getenv("STT_BACKEND", "cloud").lower(),
            local_model=os.getenv("LOCAL_STT_MODEL", "base"),
            local_device=os.getenv("LOCAL_STT_DEVICE", "cpu"),
            local_compute_type=os.getenv("LOCAL_STT_COMPUTE_TYPE", "int8"),
            local_beam_size=int(os.getenv("LOCAL_STT_BEAM_SIZE", "1")),
        )
        if self.config.backend not in {"cloud", "local", "hybrid"}:
            raise ValueError("STT_BACKEND must be 'cloud', 'local', or 'hybrid'")
        self._local_model = None
        self._local_lock = threading.Lock()
        if client is not None:
            self.client = client
        else:
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not configured")
            self.client = OpenAI(api_key=api_key, max_retries=0)

    def transcribe(self, wav_data: bytes) -> str:
        if not wav_data:
            raise ValueError("wav_data must not be empty")

        if self.config.backend in {"local", "hybrid"}:
            try:
                local_text = self._transcribe_local(wav_data)
                if local_text or self.config.backend == "local":
                    return local_text
            except Exception:
                if self.config.backend == "local":
                    raise

        request: dict[str, Any] = {
            "model": self.config.model,
            "file": ("command.wav", wav_data, "audio/wav"),
            "prompt": self.config.prompt,
            "timeout": self.config.timeout_seconds,
        }
        if self.config.model.startswith("gpt-transcribe"):
            request["extra_body"] = {
                "languages": [self.config.language],
                "keywords": list(self.config.keywords),
            }
        else:
            request["language"] = self.config.language

        transcript = self.client.audio.transcriptions.create(**request)
        text = transcript if isinstance(transcript, str) else transcript.text
        return text.strip()

    def _transcribe_local(self, wav_data: bytes) -> str:
        """Run a lazy faster-whisper model on the completed WAV buffer."""
        try:
            import numpy as np
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "local STT requires faster-whisper; install it in the active venv"
            ) from exc
        with wave.open(io.BytesIO(wav_data), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        if sample_width != 2:
            raise ValueError("local STT currently requires 16-bit PCM WAV")
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        with self._local_lock:
            if self._local_model is None:
                self._local_model = WhisperModel(
                    self.config.local_model,
                    device=self.config.local_device,
                    compute_type=self.config.local_compute_type,
                )
            segments, _ = self._local_model.transcribe(
                audio,
                language=self.config.language,
                beam_size=self.config.local_beam_size,
                best_of=self.config.local_beam_size,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=False,
                initial_prompt=self.config.prompt,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()

    def speech2text(self, wav_data: bytes | None = None) -> str:
        """Legacy method name; recording is intentionally owned by MicController."""
        if wav_data is None:
            raise ValueError(
                "speech2text() now requires WAV bytes. Use EndpointRecorder first."
            )
        return self.transcribe(wav_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe an existing WAV file")
    parser.add_argument("wav_file", type=Path)
    args = parser.parse_args()
    print(STT().transcribe(args.wav_file.read_bytes()))
