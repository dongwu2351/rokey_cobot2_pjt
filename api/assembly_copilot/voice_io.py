from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
LLM_ROOT = ROOT / "llm"
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))


def _verbose_logs() -> bool:
    return os.getenv("DUME_VERBOSE_LOGS", "false").lower() in {"1", "true", "yes", "on"}


def _debug(message: str) -> None:
    if _verbose_logs():
        print(message, flush=True)


class _TranscriptOnlyRouter:
    """Satisfy VoiceCommandPipeline while deferring semantics to the copilot."""

    def parse_command(self, utterance, context):
        from VoiceProcessing.command_models import Ambiguity, CommandResult, Decision
        return CommandResult(
            decision=Decision.REJECT,
            actions=(),
            ambiguity=Ambiguity.VISION_GROUNDING_REQUIRED,
            clarification_question=None,
            route="SAFETY",
            raw_utterance=utterance,
            grounding_query=utterance,
        )


class VoiceQuestionListener:
    def __init__(self, callback: Callable[[str], None], *, no_wake: bool = False,
                 device_index: int | None = None, sample_rate: int | None = None,
                 preflight: bool = True, vad_threshold: float = 0.5,
                 end_silence_ms: int = 450) -> None:
        self.callback = callback
        self.no_wake = no_wake
        self._stop = threading.Event()
        from VoiceProcessing.voice_pipeline import build_pipeline
        self.pipeline = build_pipeline(SimpleNamespace(
            no_wake=no_wake, device_index=device_index, sample_rate=sample_rate,
            vad_threshold=vad_threshold, start_timeout_ms=10_000,
            max_record_ms=12_000, end_silence_ms=end_silence_ms))
        # Only STT is used; assembly routing happens in the copilot.
        self.pipeline.router = _TranscriptOnlyRouter()
        if preflight:
            self._preflight_microphone(device_index, sample_rate)
        else:
            microphone = self.pipeline.microphone
            _debug(
                f"[마이크 설정] device={device_index if device_index is not None else 'default'}, "
                f"rate={microphone.config.rate}, channels={microphone.config.channels} "
                "(실제 녹음에서 최초 개방)"
            )
        self._empty_runs = 0
        self.thread = threading.Thread(target=self._loop, daemon=True,
                                       name="assembly-voice-listener")

    def _preflight_microphone(self, device_index, sample_rate) -> None:
        microphone = self.pipeline.microphone
        try:
            microphone.open_stream()
            microphone.close_stream()
        except Exception as exc:
            raise RuntimeError(
                f"마이크를 열 수 없습니다(device_index={device_index}, "
                f"sample_rate={sample_rate or microphone.config.rate}). "
                "`python3 llm/VoiceProcessing/mic_test.py --list-devices`로 "
                f"현재 입력 장치를 다시 확인하세요: {exc}") from exc
        _debug(
            f"[마이크 준비] device={device_index if device_index is not None else 'default'}, "
            f"rate={microphone.config.rate}, channels={microphone.config.channels}"
        )

    def start(self) -> None:
        self.thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.listen_once()

    def listen_once(self) -> None:
        try:
            result = self.pipeline.run_once(wait_for_wake=not self.no_wake)
            text = (result.transcript or "").strip()
            if text:
                self._empty_runs = 0
                self.callback(text)
                timings = getattr(result, "timings", None)
                if timings is not None:
                    _debug(f"[음성 성능] recording={timings.recording_ms:.0f}ms, "
                           f"stt={timings.stt_ms:.0f}ms, total={timings.total_ms:.0f}ms")
            else:
                self._empty_runs += 1
                state = getattr(result.state, "value", str(result.state))
                if state not in {"NO_SPEECH"} or self._empty_runs == 1:
                    reason = getattr(getattr(result, "recording", None),
                                     "stop_reason", None)
                    _debug(f"[음성 결과] state={state}, stop_reason={reason}, "
                           f"error={getattr(result, 'error', None)}")
        except Exception as exc:
            _debug(f"[음성 입력 실패] {type(exc).__name__}: {exc}")
            self._stop.wait(1.0)

    def close(self) -> None:
        self._stop.set()
        microphone = getattr(self.pipeline, "microphone", None)
        close = getattr(microphone, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def _voice_process_main(output_queue, stop_event, pause_event, no_wake,
                        device_index, sample_rate, vad_threshold,
                        end_silence_ms) -> None:
    """Keep one audio runtime alive; Pulse owns all capture sessions here."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if not _verbose_logs():
        # ALSA/PortAudio diagnostics are native writes to fd 2 and cannot be
        # filtered with Python logging. Keep the user console conversation-only.
        stderr_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
    def publish(text):
        if not pause_event.is_set():
            output_queue.put(text)

    listener = VoiceQuestionListener(
        publish, no_wake=no_wake,
        device_index=device_index, sample_rate=sample_rate,
        # This PortAudio/Pulse combination can segfault when a successful
        # preflight open/close is immediately followed by a second open.
        preflight=False, vad_threshold=vad_threshold,
        end_silence_ms=end_silence_ms)
    try:
        while not stop_event.is_set():
            if pause_event.is_set():
                stop_event.wait(0.05)
                continue
            listener.listen_once()
    finally:
        listener.close()


class VoiceQuestionProcess:
    """Process-isolated microphone frontend for the integrated RGB-D runtime."""

    def __init__(self, *, no_wake: bool = False, device_index: int | None = None,
                 sample_rate: int | None = None, vad_threshold: float = 0.5,
                 end_silence_ms: int = 450) -> None:
        context = mp.get_context("spawn")
        self._context = context
        self.questions = context.Queue()
        self._stop = context.Event()
        self._pause = context.Event()
        self.process = context.Process(
            target=_voice_process_main,
            args=(self.questions, self._stop, self._pause, no_wake,
                  device_index, sample_rate,
                  vad_threshold, end_silence_ms),
            daemon=True, name="dume-voice-capture")

    def start(self) -> None:
        self.process.start()
        _debug(f"[음성 프로세스] 시작됨(pid={self.process.pid})")

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def close(self) -> None:
        self._stop.set()
        self.process.join(timeout=3.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2.0)
        self.process.close()
        self.questions.close()
        self.questions.join_thread()


class CopilotSpeech:
    def __init__(self, enabled: bool, *, listener=None) -> None:
        from speech_manager import PRIO_CHAT, SpeechManager
        self.priority = PRIO_CHAT
        # Keep capture active during playback. SpeechManager suppresses its own
        # echo, while a distinct user utterance cancels playback (barge-in).
        self.manager = SpeechManager(enabled=enabled, warmup=False)

    def say(self, text: str) -> None:
        # Keep the full answer on screen, but do not monopolize the microphone
        # with a long spoken manual paragraph.
        spoken = text.strip()
        if len(spoken) > 150:
            candidate = spoken[:150]
            boundary = max(candidate.rfind("다."), candidate.rfind("요."),
                           candidate.rfind("."))
            spoken = candidate[:boundary + 1] if boundary > 80 else candidate + "…"
        self.manager.say(spoken, priority=self.priority)

    def accept_transcript(self, text: str) -> bool:
        if self.manager.is_likely_echo(text):
            _debug(f"[TTS 에코 무시] {text}")
            return False
        if self.manager.is_playing():
            self.manager.cancel("사용자 끼어들기")
        return True

    def close(self) -> None:
        self.manager.close()
