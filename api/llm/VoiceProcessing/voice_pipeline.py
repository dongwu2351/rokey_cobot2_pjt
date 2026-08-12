from __future__ import annotations

import argparse
import json
import os
import threading
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from dotenv import load_dotenv

from .MicController import MicConfig, MicController
from .STT import ENV_PATH, STT
from .command_models import CommandContext, CommandResult, Decision, Intent
from .command_router import CommandConfig, CommandRouter, LLMCommandParser
from .vad import (
    AdaptiveEnergyVAD,
    EndpointRecorder,
    RecordingResult,
    StopReason,
    VADConfig,
    build_default_vad,
)
from .wakeup_word import WakeupWord


def _suspend_aware_clock() -> float:
    """Monotonic time that includes system suspend on Linux when available."""
    clock_id = getattr(time, "CLOCK_BOOTTIME", None)
    if clock_id is not None:
        try:
            return time.clock_gettime(clock_id)
        except (OSError, ValueError):
            pass
    return time.monotonic()


class PipelineState(str, Enum):
    WAIT_WAKE = "WAIT_WAKE"
    WAIT_SPEECH = "WAIT_SPEECH"
    TRANSCRIBE = "TRANSCRIBE"
    INTERPRET = "INTERPRET"
    COMMAND_READY = "COMMAND_READY"
    COMMAND_NOT_READY = "COMMAND_NOT_READY"
    NO_SPEECH = "NO_SPEECH"
    TRUNCATED = "TRUNCATED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class PipelineTimings:
    wake_wait_ms: float = 0.0
    recording_ms: float = 0.0
    stt_ms: float = 0.0
    command_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(frozen=True)
class PipelineResult:
    state: PipelineState
    transcript: str | None
    recording: RecordingResult | None
    command: CommandResult | None
    timings: PipelineTimings
    error: str | None = None
    context_valid_until_ms: int | None = None
    _context_was_fresh: bool = False
    _context_deadline_expiry_clock: float | None = None
    _expiry_clock: Callable[[], float] = field(
        default=_suspend_aware_clock,
        repr=False,
        compare=False,
    )
    _wall_clock: Callable[[], float] = field(
        default=time.time,
        repr=False,
        compare=False,
    )
    _expired_latch: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
        compare=False,
    )

    @property
    def safety_context_fresh(self) -> bool:
        if self._expired_latch.is_set():
            return False
        if (
            not self._context_was_fresh
            or self.context_valid_until_ms is None
            or self._context_deadline_expiry_clock is None
        ):
            return False
        expired = (
            round(self._wall_clock() * 1_000) > self.context_valid_until_ms
            or self._expiry_clock() > self._context_deadline_expiry_clock
        )
        if expired:
            self._expired_latch.set()
            return False
        return True

    @property
    def executable(self) -> bool:
        if (
            self.state != PipelineState.COMMAND_READY
            or self.command is None
            or not isinstance(self.command, CommandResult)
            or self.command.decision != Decision.READY
            or not self.command.actions
        ):
            return False

        stop_actions = [
            action for action in self.command.actions if action.intent == Intent.STOP
        ]
        if stop_actions:
            action = stop_actions[0]
            return (
                len(self.command.actions) == 1
                and action.object is None
                and action.destination is None
                and action.object_query is None
                and action.resolved_object_id is None
            )

        has_snapshot_identity = (
            self.command.snapshot_revision is not None
            and self.command.snapshot_timestamp_ms is not None
        )
        allowed_motion_intents = {Intent.FETCH, Intent.MOVE, Intent.PLACE}
        resolved_ids = [
            action.resolved_object_id for action in self.command.actions
        ]
        return (
            self.safety_context_fresh
            and has_snapshot_identity
            and all(action.intent in allowed_motion_intents for action in self.command.actions)
            and all(action.object is not None for action in self.command.actions)
            and all(action.resolved_object_id is not None for action in self.command.actions)
            and len(resolved_ids) == len(set(resolved_ids))
            and all(
                (action.intent == Intent.FETCH and action.destination is None)
                or (
                    action.intent in (Intent.MOVE, Intent.PLACE)
                    and action.destination is not None
                )
                for action in self.command.actions
            )
        )

    def to_dict(self, *, include_transcript: bool = True) -> dict[str, Any]:
        recording = None
        if self.recording is not None:
            recording = {
                "duration_ms": self.recording.duration_ms,
                "speech_ms": self.recording.speech_ms,
                "stop_reason": self.recording.stop_reason.value,
                "speech_detected": self.recording.speech_detected,
            }
        command = self.command.model_dump(mode="json") if self.command is not None else None
        if command is not None and not include_transcript:
            command["raw_utterance"] = "[REDACTED]"
        return {
            "state": self.state.value,
            "executable": self.executable,
            "safety_context_fresh": self.safety_context_fresh,
            "context_valid_until_ms": self.context_valid_until_ms,
            "transcript": self.transcript if include_transcript else None,
            "recording": recording,
            "command": command,
            "timings": {
                "wake_wait_ms": self.timings.wake_wait_ms,
                "recording_ms": self.timings.recording_ms,
                "stt_ms": self.timings.stt_ms,
                "command_ms": self.timings.command_ms,
                "total_ms": self.timings.total_ms,
            },
            "error": self.error,
        }


class _BufferedFrameReader:
    """Re-block wake-detector remainder and microphone data into VAD frames."""

    def __init__(
        self,
        initial_pcm: bytes,
        read_frame: Callable[[], bytes],
        frame_bytes: int,
    ) -> None:
        self.buffer = bytearray(initial_pcm)
        self.read_frame = read_frame
        self.frame_bytes = frame_bytes

    def __call__(self) -> bytes:
        while len(self.buffer) < self.frame_bytes:
            self.buffer.extend(self.read_frame())
        frame = bytes(self.buffer[: self.frame_bytes])
        del self.buffer[: self.frame_bytes]
        return frame


class VoiceCommandPipeline:
    """Wake word -> VAD endpoint -> STT -> deterministic/LLM command route."""

    def __init__(
        self,
        *,
        microphone: Any,
        recorder: EndpointRecorder,
        transcriber: Any,
        router: Any,
        wake_detector: Any | None = None,
        clock: Callable[[], float] = time.perf_counter,
        wall_clock: Callable[[], float] = time.time,
        expiry_clock: Callable[[], float] = _suspend_aware_clock,
        max_context_age_ms: int | None = None,
        max_future_skew_ms: int | None = None,
        partial_callback: Callable[[bytes, int], None] | None = None,
        partial_interval_ms: int = 500,
    ) -> None:
        self.microphone = microphone
        self.recorder = recorder
        self.transcriber = transcriber
        self.router = router
        self.wake_detector = wake_detector
        self.clock = clock
        self.wall_clock = wall_clock
        self.expiry_clock = expiry_clock
        router_config = getattr(router, "config", None)
        self.max_context_age_ms = (
            max_context_age_ms
            if max_context_age_ms is not None
            else getattr(router_config, "max_snapshot_age_ms", 3_000)
        )
        self.max_future_skew_ms = (
            max_future_skew_ms
            if max_future_skew_ms is not None
            else getattr(router_config, "max_future_skew_ms", 250)
        )
        if self.max_context_age_ms <= 0:
            raise ValueError("max_context_age_ms must be positive")
        if self.max_future_skew_ms < 0:
            raise ValueError("max_future_skew_ms cannot be negative")
        if partial_interval_ms <= 0:
            raise ValueError("partial_interval_ms must be positive")
        self.partial_callback = partial_callback
        self.partial_interval_ms = partial_interval_ms

    def run_once(
        self,
        *,
        wait_for_wake: bool = True,
        context: CommandContext | Mapping[str, Any] | None = None,
        context_provider: Callable[
            [], Mapping[str, Any] | CommandContext | None
        ]
        | None = None,
    ) -> PipelineResult:
        total_started = self.clock()
        wake_ms = recording_ms = stt_ms = command_ms = 0.0
        recording: RecordingResult | None = None
        transcript: str | None = None
        command: CommandResult | None = None
        owns_stream = not bool(getattr(self.microphone, "is_open", False))
        owned_stream_open = False

        try:
            if not owns_stream:
                raise RuntimeError(
                    "run_once requires a closed microphone so the pipeline can "
                    "enforce half-duplex ownership"
                )
            if owns_stream:
                self.microphone.open_stream()
                owned_stream_open = True

            if wait_for_wake:
                if self.wake_detector is None:
                    raise RuntimeError("wait_for_wake=True requires a wake detector")
                wake_started = self.clock()
                while True:
                    frame = self.microphone.read_frame()
                    if self.wake_detector.process(
                        frame,
                        sample_rate=self.microphone.config.rate,
                    ):
                        break
                wake_ms = (self.clock() - wake_started) * 1_000

            initial_pcm = b""
            if wait_for_wake and hasattr(self.wake_detector, "drain_remainder"):
                initial_pcm = self.wake_detector.drain_remainder(
                    sample_rate=self.microphone.config.rate
                )
            frame_reader = _BufferedFrameReader(
                initial_pcm,
                self.microphone.read_frame,
                self.recorder.bytes_per_frame,
            )
            recording_started = self.clock()
            if self.partial_callback is None:
                recording = self.recorder.record(frame_reader)
            else:
                recording = self.recorder.record(
                    frame_reader,
                    on_partial=self.partial_callback,
                    partial_interval_ms=self.partial_interval_ms,
                )
            recording_ms = (self.clock() - recording_started) * 1_000

            # The network and command stages are explicitly half-duplex. Closing
            # here prevents PortAudio from accumulating stale input while these
            # blocking calls run. A close failure stops the command before STT.
            if owned_stream_open:
                self.microphone.close_stream()
                owned_stream_open = False

            if not recording.speech_detected:
                return self._result(
                    PipelineState.NO_SPEECH,
                    total_started,
                    wake_ms,
                    recording_ms,
                    stt_ms,
                    command_ms,
                    recording=recording,
                )
            if recording.stop_reason == StopReason.MAX_DURATION:
                return self._result(
                    PipelineState.TRUNCATED,
                    total_started,
                    wake_ms,
                    recording_ms,
                    stt_ms,
                    command_ms,
                    recording=recording,
                    error="MAX_DURATION",
                )

            stt_started = self.clock()
            transcript = self.transcriber.transcribe(recording.to_wav_bytes())
            stt_ms = (self.clock() - stt_started) * 1_000

            command_started = self.clock()
            fresh_context = context_provider() if context_provider else context
            parsed_command = self.router.parse_command(transcript, fresh_context)
            if not isinstance(parsed_command, CommandResult):
                raise TypeError("router must return CommandResult")
            command = parsed_command
            command_ms = (self.clock() - command_started) * 1_000
            stop_only = (
                len(command.actions) == 1
                and command.actions[0].intent == Intent.STOP
            )
            now_ms = round(self.wall_clock() * 1_000)
            snapshot_timestamp_ms = getattr(
                command,
                "snapshot_timestamp_ms",
                None,
            )
            snapshot_revision = getattr(command, "snapshot_revision", None)
            snapshot_age_ms = (
                now_ms - snapshot_timestamp_ms
                if snapshot_timestamp_ms is not None
                else None
            )
            has_fresh_context = (
                context_provider is not None
                and fresh_context is not None
                and snapshot_revision is not None
                and snapshot_age_ms is not None
                and -self.max_future_skew_ms
                <= snapshot_age_ms
                <= self.max_context_age_ms
            )
            context_valid_until_ms = (
                snapshot_timestamp_ms + self.max_context_age_ms
                if has_fresh_context
                else None
            )
            context_deadline_expiry_clock = (
                self.expiry_clock()
                + (self.max_context_age_ms - snapshot_age_ms) / 1_000
                if has_fresh_context
                else None
            )
            state = (
                PipelineState.COMMAND_READY
                if command.decision == Decision.READY
                and (stop_only or has_fresh_context)
                else PipelineState.COMMAND_NOT_READY
            )
            return self._result(
                state,
                total_started,
                wake_ms,
                recording_ms,
                stt_ms,
                command_ms,
                transcript=transcript,
                recording=recording,
                command=command,
                safety_context_fresh=has_fresh_context,
                context_valid_until_ms=context_valid_until_ms,
                context_deadline_expiry_clock=context_deadline_expiry_clock,
            )
        except Exception as exc:
            return self._result(
                PipelineState.ERROR,
                total_started,
                wake_ms,
                recording_ms,
                stt_ms,
                command_ms,
                transcript=transcript,
                recording=recording,
                command=command,
                error=type(exc).__name__,
            )
        finally:
            if self.wake_detector is not None:
                try:
                    self.wake_detector.reset()
                except Exception as exc:
                    warnings.warn(
                        f"wake-detector cleanup failed: {type(exc).__name__}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if owned_stream_open:
                try:
                    self.microphone.close_stream()
                    owned_stream_open = False
                except Exception as exc:
                    warnings.warn(
                        f"microphone cleanup failed: {type(exc).__name__}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

    def run_forever(
        self,
        callback: Callable[[PipelineResult], None],
        *,
        context_provider: Callable[[], Mapping[str, Any] | CommandContext | None]
        | None = None,
        max_consecutive_errors: int = 5,
        base_backoff_seconds: float = 0.25,
    ) -> None:
        consecutive_errors = 0
        try:
            # ``run_forever`` owns capture-session boundaries. Normalize a
            # caller-provided open stream to the same closed half-duplex state
            # used between every iteration.
            if bool(getattr(self.microphone, "is_open", False)):
                self.microphone.close_stream()
            while True:
                result = self.run_once(
                    wait_for_wake=True,
                    context_provider=context_provider,
                )
                if bool(getattr(self.microphone, "is_open", False)):
                    raise RuntimeError(
                        "microphone remained open after half-duplex capture"
                    )
                callback(result)
                if result.state == PipelineState.ERROR:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        raise RuntimeError(
                            "voice pipeline stopped after consecutive errors"
                        )
                    time.sleep(
                        min(
                            4.0,
                            base_backoff_seconds * (2 ** (consecutive_errors - 1)),
                        )
                    )
                else:
                    consecutive_errors = 0
        finally:
            if bool(getattr(self.microphone, "is_open", False)):
                try:
                    self.microphone.close_stream()
                except Exception as exc:
                    warnings.warn(
                        f"microphone cleanup failed: {type(exc).__name__}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

    def _result(
        self,
        state: PipelineState,
        total_started: float,
        wake_ms: float,
        recording_ms: float,
        stt_ms: float,
        command_ms: float,
        *,
        transcript: str | None = None,
        recording: RecordingResult | None = None,
        command: CommandResult | None = None,
        error: str | None = None,
        safety_context_fresh: bool = False,
        context_valid_until_ms: int | None = None,
        context_deadline_expiry_clock: float | None = None,
    ) -> PipelineResult:
        timings = PipelineTimings(
            wake_wait_ms=round(wake_ms, 3),
            recording_ms=round(recording_ms, 3),
            stt_ms=round(stt_ms, 3),
            command_ms=round(command_ms, 3),
            total_ms=round((self.clock() - total_started) * 1_000, 3),
        )
        return PipelineResult(
            state=state,
            transcript=transcript,
            recording=recording,
            command=command,
            timings=timings,
            error=error,
            context_valid_until_ms=context_valid_until_ms,
            _context_was_fresh=safety_context_fresh,
            _context_deadline_expiry_clock=context_deadline_expiry_clock,
            _expiry_clock=self.expiry_clock,
            _wall_clock=self.wall_clock,
        )


def build_pipeline(args: argparse.Namespace) -> VoiceCommandPipeline:
    load_dotenv(ENV_PATH)
    rate = args.sample_rate or int(os.getenv("MIC_SAMPLE_RATE", "16000"))
    device_index = args.device_index
    if device_index is None and os.getenv("MIC_DEVICE_INDEX"):
        device_index = int(os.environ["MIC_DEVICE_INDEX"])

    mic_config = MicConfig(rate=rate, device_index=device_index)
    microphone = MicController(mic_config)
    if rate == 16_000:
        detector = build_default_vad(threshold=args.vad_threshold)
    else:
        detector = AdaptiveEnergyVAD()
    vad_config = VADConfig(
        frame_ms=mic_config.frame_ms,
        start_timeout_ms=args.start_timeout_ms,
        max_record_ms=args.max_record_ms,
        end_silence_ms=args.end_silence_ms,
    )
    recorder = EndpointRecorder(
        detector,
        sample_rate=mic_config.rate,
        config=vad_config,
    )
    wake_detector = None if args.no_wake else WakeupWord(input_rate=rate)
    transcriber = STT()
    command_config = CommandConfig.from_env()
    router = CommandRouter(
        config=command_config,
        llm_parser=LLMCommandParser(
            config=command_config,
            client=transcriber.client,
        ),
    )
    return VoiceCommandPipeline(
        microphone=microphone,
        recorder=recorder,
        transcriber=transcriber,
        router=router,
        wake_detector=wake_detector,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ROKEY low-latency voice pipeline")
    parser.add_argument("--no-wake", action="store_true", help="record immediately")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--start-timeout-ms", type=int, default=2_000)
    parser.add_argument("--max-record-ms", type=int, default=10_000)
    parser.add_argument("--end-silence-ms", type=int, default=600)
    parser.add_argument(
        "--show-transcript",
        action="store_true",
        help="include raw speech text in console output",
    )
    parser.add_argument(
        "--speak-clarifications",
        action="store_true",
        help="speak CLARIFY questions after the microphone stream is closed",
    )
    args = parser.parse_args()

    pipeline = build_pipeline(args)
    synthesizer = None
    if args.speak_clarifications:
        from .TTS import SpeechSynthesizer

        synthesizer = SpeechSynthesizer(client=pipeline.transcriber.client)

    def print_result(result: PipelineResult) -> None:
        print(
            json.dumps(
                result.to_dict(include_transcript=args.show_transcript),
                ensure_ascii=False,
                indent=2,
            )
        )
        question = (
            result.command.clarification_question
            if result.command is not None
            and result.command.decision == Decision.CLARIFY
            else None
        )
        if synthesizer is not None and question:
            try:
                synthesizer.speak(question)
            except Exception as exc:
                warnings.warn(
                    f"clarification TTS failed: {type(exc).__name__}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    try:
        if args.continuous:
            if args.no_wake:
                parser.error("--continuous requires wake-word mode")
            pipeline.run_forever(print_result)
        else:
            print_result(pipeline.run_once(wait_for_wake=not args.no_wake))
    except KeyboardInterrupt:
        print("\n종료합니다.")


if __name__ == "__main__":
    main()
