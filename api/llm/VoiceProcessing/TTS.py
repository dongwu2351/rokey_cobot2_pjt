from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pyaudio
from dotenv import load_dotenv
from openai import OpenAI


ENV_PATH = Path(__file__).with_name(".env")
AI_VOICE_DISCLOSURE = "지금부터 들리는 안내 음성은 인공지능이 생성한 음성입니다."


@dataclass(frozen=True)
class TTSConfig:
    model: str = "gpt-4o-mini-tts"
    voice: str = "marin"
    instructions: str = (
        "한국어로 짧고 또렷하게 말하세요. 협동로봇 작업자가 즉시 이해할 수 있도록 "
        "차분한 속도로 질문하세요."
    )
    timeout_seconds: float = 8.0
    speed: float = 1.0
    playback_backend: str = "auto"
    streaming: bool = False


class SpeechSynthesizer:
    """Generate short spoken robot responses and optionally play WAV bytes."""

    def __init__(
        self,
        openai_api_key: str | None = None,
        *,
        config: TTSConfig | None = None,
        client: Any | None = None,
        audio_factory: Callable[[], pyaudio.PyAudio] = pyaudio.PyAudio,
    ) -> None:
        load_dotenv(ENV_PATH)
        self.config = config or TTSConfig(
            model=os.getenv("TTS_MODEL", "gpt-4o-mini-tts"),
            voice=os.getenv("TTS_VOICE", "marin"),
            instructions=os.getenv(
                "TTS_INSTRUCTIONS",
                "한국어로 짧고 또렷하게 말하세요. 협동로봇 작업자가 즉시 이해할 수 "
                "있도록 차분하고 자연스러운 속도로 안내하세요."),
            speed=float(os.getenv("TTS_SPEED", "1.0")),
            playback_backend=os.getenv("TTS_PLAYBACK_BACKEND", "auto"),
            streaming=os.getenv("TTS_STREAMING", "0").lower() in {"1", "true", "yes"},
        )
        if self.config.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 0.25 <= self.config.speed <= 4.0:
            raise ValueError("TTS speed must be between 0.25 and 4.0")
        self._audio_factory = audio_factory
        self._speak_lock = threading.Lock()
        self._playback_backend = self.config.playback_backend
        if client is not None:
            self.client = client
        else:
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not configured")
            self.client = OpenAI(api_key=api_key, max_retries=0)

    def synthesize(self, text: str) -> bytes:
        spoken_text = text.strip()
        if not spoken_text:
            raise ValueError("text must not be empty")
        response = self.client.audio.speech.create(
            model=self.config.model,
            voice=self.config.voice,
            input=spoken_text,
            instructions=self.config.instructions,
            speed=self.config.speed,
            response_format="wav",
            timeout=self.config.timeout_seconds,
        )
        return response.read()

    def play_wav(self, wav_data: bytes) -> None:
        if not wav_data:
            raise ValueError("wav_data must not be empty")
        backend = self._playback_backend
        if backend == "auto":
            backend = "aplay" if shutil.which("aplay") else "pyaudio"
        if backend == "aplay":
            try:
                subprocess.run(
                    ["aplay", "-q"],
                    input=wav_data,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=max(5.0, self.config.timeout_seconds + 4.0),
                )
                return
            except (FileNotFoundError, subprocess.TimeoutExpired):
                # Fall back to the existing PyAudio path if aplay is absent or
                # the sound server does not respond.
                pass
        with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
            audio = self._audio_factory()
            stream = None
            try:
                stream = audio.open(
                    format=audio.get_format_from_width(wav_file.getsampwidth()),
                    channels=wav_file.getnchannels(),
                    rate=wav_file.getframerate(),
                    output=True,
                )
                while chunk := wav_file.readframes(4096):
                    stream.write(chunk)
            finally:
                if stream is not None:
                    try:
                        stream.stop_stream()
                    finally:
                        stream.close()
                audio.terminate()

    def speak(self, text: str) -> None:
        # Prevent overlapping playback/restarts when two runtime callbacks
        # arrive close together.
        with self._speak_lock:
            if self.config.streaming:
                self.stream_speak(text)
            else:
                self.play_wav(self.synthesize(text))

    def stream_speak(self, text: str) -> None:
        """Play PCM chunks as they arrive from the Speech API."""
        spoken_text = text.strip()
        if not spoken_text:
            raise ValueError("text must not be empty")
        audio = self._audio_factory()
        stream = None
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24_000,
                output=True,
            )
            with self.client.audio.speech.with_streaming_response.create(
                model=self.config.model,
                voice=self.config.voice,
                input=spoken_text,
                instructions=self.config.instructions,
                speed=self.config.speed,
                response_format="pcm",
                timeout=self.config.timeout_seconds,
            ) as response:
                for chunk in response.iter_bytes(chunk_size=4096):
                    if chunk:
                        stream.write(chunk)
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                finally:
                    stream.close()
            audio.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a short Korean TTS response")
    parser.add_argument("text")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--disclosure", action="store_true")
    args = parser.parse_args()

    synthesizer = SpeechSynthesizer()
    text = f"{AI_VOICE_DISCLOSURE} {args.text}" if args.disclosure else args.text
    wav_data = synthesizer.synthesize(text)
    if args.output:
        args.output.write_bytes(wav_data)
    if args.play or not args.output:
        synthesizer.play_wav(wav_data)


if __name__ == "__main__":
    main()
