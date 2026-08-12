from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Callable

import pyaudio


@dataclass(frozen=True)
class MicConfig:
    rate: int = 16_000
    channels: int = 1
    frame_ms: int = 30
    record_seconds: float = 5.0  # legacy fixed-duration API only
    fmt: int = pyaudio.paInt16
    device_index: int | None = None

    @property
    def chunk(self) -> int:
        samples = self.rate * self.frame_ms
        if samples % 1_000:
            raise ValueError("rate * frame_ms must produce a whole sample count")
        return samples // 1_000

    @property
    def buffer_size(self) -> int:
        """Legacy name: openWakeWord consumes 80 ms at 16 kHz."""
        return self.rate * 80 // 1_000


class MicController:
    """Own one PyAudio stream for a wake-word/VAD capture session.

    The pipeline closes that session before remote STT and command inference so
    PortAudio cannot accumulate stale audio while no consumer is reading it.
    """

    def __init__(
        self,
        config: MicConfig | None = None,
        *,
        audio_factory: Callable[[], pyaudio.PyAudio] = pyaudio.PyAudio,
    ) -> None:
        self.config = config or MicConfig()
        self._audio_factory = audio_factory
        self.audio: pyaudio.PyAudio | None = None
        self.stream = None
        self.sample_width = 2

    @property
    def is_open(self) -> bool:
        return self.stream is not None

    def open_stream(self) -> None:
        if self.is_open:
            return
        self.audio = self._audio_factory()
        self.sample_width = self.audio.get_sample_size(self.config.fmt)
        stream_args = {
            "format": self.config.fmt,
            "channels": self.config.channels,
            "rate": self.config.rate,
            "input": True,
            "frames_per_buffer": self.config.chunk,
        }
        if self.config.device_index is not None:
            stream_args["input_device_index"] = self.config.device_index
        try:
            self.stream = self.audio.open(**stream_args)
        except Exception:
            self.audio.terminate()
            self.audio = None
            raise

    def close_stream(self) -> None:
        stream, audio = self.stream, self.audio
        self.stream = None
        self.audio = None
        try:
            if stream is not None:
                try:
                    if stream.is_active():
                        stream.stop_stream()
                finally:
                    stream.close()
        finally:
            if audio is not None:
                audio.terminate()

    def read_frame(self) -> bytes:
        if self.stream is None:
            raise RuntimeError("microphone stream is not open")
        return self.stream.read(
            self.config.chunk,
            exception_on_overflow=False,
        )

    def record_audio(self, duration_seconds: float | None = None) -> bytes:
        """Backward-compatible fixed recording using the current stream.

        New code should use ``EndpointRecorder`` so recording ends on silence.
        """
        owns_stream = not self.is_open
        if owns_stream:
            self.open_stream()
        duration = duration_seconds or self.config.record_seconds
        frame_count = max(1, round(duration * 1_000 / self.config.frame_ms))
        try:
            pcm = b"".join(self.read_frame() for _ in range(frame_count))
            return self.to_wav_bytes(pcm)
        finally:
            if owns_stream:
                self.close_stream()

    def to_wav_bytes(self, pcm: bytes) -> bytes:
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(self.config.channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.config.rate)
            wav_file.writeframes(pcm)
        return wav_io.getvalue()

    def __enter__(self) -> "MicController":
        self.open_stream()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close_stream()
