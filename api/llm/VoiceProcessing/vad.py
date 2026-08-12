from __future__ import annotations

import math
import wave
from collections import deque
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Callable, Protocol
import warnings

import numpy as np


class VoiceActivityDetector(Protocol):
    def reset(self) -> None: ...

    def is_speech(self, pcm: bytes) -> bool: ...


class StopReason(str, Enum):
    END_SILENCE = "end_silence"
    START_TIMEOUT = "start_timeout"
    TOO_SHORT = "too_short"
    MAX_DURATION = "max_duration"


@dataclass(frozen=True)
class VADConfig:
    frame_ms: int = 30
    start_timeout_ms: int = 2_000
    max_record_ms: int = 10_000
    onset_frames: int = 2
    end_silence_ms: int = 600
    pre_roll_ms: int = 180
    trailing_silence_ms: int = 180
    min_speech_ms: int = 180

    def __post_init__(self) -> None:
        if self.frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20, or 30")
        if self.onset_frames < 1:
            raise ValueError("onset_frames must be at least 1")
        if self.start_timeout_ms < self.frame_ms:
            raise ValueError("start_timeout_ms must be at least one frame")
        if self.max_record_ms < self.frame_ms:
            raise ValueError("max_record_ms must be at least one frame")
        if self.trailing_silence_ms > self.end_silence_ms:
            raise ValueError("trailing_silence_ms cannot exceed end_silence_ms")
        if min(
            self.pre_roll_ms,
            self.trailing_silence_ms,
            self.end_silence_ms,
            self.min_speech_ms,
        ) < 0:
            raise ValueError("VAD durations cannot be negative")
        if math.ceil(self.pre_roll_ms / self.frame_ms) < self.onset_frames:
            raise ValueError("pre-roll must retain every onset frame")


@dataclass(frozen=True)
class RecordingResult:
    pcm: bytes
    sample_rate: int
    stop_reason: StopReason
    speech_detected: bool
    speech_ms: int
    frame_ms: int
    sample_width: int = 2
    channels: int = 1

    @property
    def duration_ms(self) -> int:
        if not self.pcm:
            return 0
        frame_count = len(self.pcm) // (self.sample_width * self.channels)
        return round(frame_count / self.sample_rate * 1_000)

    def to_wav_bytes(self) -> bytes:
        output = BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(self.pcm)
        return output.getvalue()


class AdaptiveEnergyVAD:
    """Small dependency-free VAD used when the Silero asset is unavailable.

    The noise estimate follows silence and slowly catches up to sustained moderate
    machinery noise. Loud close-mic speech does not raise its own threshold.
    """

    def __init__(
        self,
        *,
        absolute_floor_dbfs: float = -46.0,
        speech_margin_db: float = 10.0,
        noise_alpha: float = 0.95,
        initial_noise_dbfs: float = -60.0,
        loud_speech_dbfs: float = -30.0,
        noise_rise_db_per_frame: float = 2.0,
    ) -> None:
        self.absolute_floor_dbfs = absolute_floor_dbfs
        self.speech_margin_db = speech_margin_db
        self.noise_alpha = noise_alpha
        self.initial_noise_dbfs = initial_noise_dbfs
        self.loud_speech_dbfs = loud_speech_dbfs
        self.noise_rise_db_per_frame = noise_rise_db_per_frame
        self.noise_dbfs = initial_noise_dbfs

    def reset(self) -> None:
        self.noise_dbfs = self.initial_noise_dbfs

    @staticmethod
    def _dbfs(pcm: bytes) -> float:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(samples * samples)))
        if rms < 1.0:
            return -120.0
        return 20.0 * math.log10(rms / 32768.0)

    def is_speech(self, pcm: bytes) -> bool:
        level = self._dbfs(pcm)
        threshold = max(
            self.absolute_floor_dbfs,
            self.noise_dbfs + self.speech_margin_db,
        )
        speech = level > threshold
        if not speech:
            self.noise_dbfs = (
                self.noise_alpha * self.noise_dbfs
                + (1.0 - self.noise_alpha) * level
            )
        elif level < self.loud_speech_dbfs:
            # A moderate, sustained level immediately after reset is usually
            # background machinery rather than close-mic speech. Let the noise
            # floor catch up, but never faster than a bounded amount per frame.
            target_noise = level - self.speech_margin_db
            self.noise_dbfs = min(
                target_noise,
                self.noise_dbfs + self.noise_rise_db_per_frame,
            )
        return speech


class SileroVAD:
    """Adapter for the Silero model distributed with openWakeWord."""

    def __init__(self, *, threshold: float = 0.5, model_path: Path | None = None):
        from openwakeword.vad import VAD

        kwargs = {"model_path": str(model_path)} if model_path is not None else {}
        self.model = VAD(**kwargs)
        self.threshold = threshold

    def reset(self) -> None:
        self.model.reset_states()

    def is_speech(self, pcm: bytes) -> bool:
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return False
        score = float(self.model.predict(samples, frame_size=len(samples)))
        return score >= self.threshold


def build_default_vad(*, threshold: float = 0.5) -> VoiceActivityDetector:
    try:
        import openwakeword

        model_path = (
            Path(openwakeword.__file__).resolve().parent
            / "resources"
            / "models"
            / "silero_vad.onnx"
        )
        if model_path.exists():
            return SileroVAD(threshold=threshold, model_path=model_path)
    except (ImportError, OSError):
        pass

    warnings.warn(
        "Silero VAD asset was not found; using AdaptiveEnergyVAD. "
        "Run the one-time openWakeWord model download for higher accuracy.",
        RuntimeWarning,
        stacklevel=2,
    )
    return AdaptiveEnergyVAD()


class EndpointRecorder:
    """Collect frames until speech ends, without knowing about microphone APIs."""

    def __init__(
        self,
        detector: VoiceActivityDetector,
        *,
        sample_rate: int = 16_000,
        config: VADConfig | None = None,
    ) -> None:
        self.detector = detector
        self.sample_rate = sample_rate
        self.config = config or VADConfig()
        expected_samples = sample_rate * self.config.frame_ms
        if expected_samples % 1_000:
            raise ValueError("sample_rate * frame_ms must produce a whole sample count")
        self.samples_per_frame = expected_samples // 1_000
        self.bytes_per_frame = self.samples_per_frame * 2

    @staticmethod
    def _ceil_frames(duration_ms: int, frame_ms: int, *, minimum: int = 1) -> int:
        return max(minimum, math.ceil(duration_ms / frame_ms))

    def record(
        self,
        read_frame: Callable[[], bytes],
        *,
        on_partial: Callable[[bytes, int], None] | None = None,
        partial_interval_ms: int = 500,
    ) -> RecordingResult:
        cfg = self.config
        self.detector.reset()

        pre_roll_frames = self._ceil_frames(cfg.pre_roll_ms, cfg.frame_ms)
        start_timeout_frames = self._ceil_frames(cfg.start_timeout_ms, cfg.frame_ms)
        max_record_frames = self._ceil_frames(cfg.max_record_ms, cfg.frame_ms)
        end_silence_frames = self._ceil_frames(cfg.end_silence_ms, cfg.frame_ms)
        trailing_frames = self._ceil_frames(
            cfg.trailing_silence_ms,
            cfg.frame_ms,
            minimum=0,
        )
        min_speech_frames = self._ceil_frames(cfg.min_speech_ms, cfg.frame_ms)

        pre_roll: deque[bytes] = deque(maxlen=pre_roll_frames)
        frames: list[bytes] = []
        onset_run = 0
        silence_run = 0
        voiced_frames = 0
        speech_started = False
        partial_frames = 0
        partial_interval_frames = self._ceil_frames(
            partial_interval_ms, cfg.frame_ms
        )

        for waiting_frame in range(start_timeout_frames):
            frame = read_frame()
            self._validate_frame(frame)
            pre_roll.append(frame)
            if self.detector.is_speech(frame):
                onset_run += 1
            else:
                onset_run = 0

            if onset_run >= cfg.onset_frames:
                frames.extend(pre_roll)
                voiced_frames = onset_run
                speech_started = True
                break
        else:
            return RecordingResult(
                pcm=b"",
                sample_rate=self.sample_rate,
                stop_reason=StopReason.START_TIMEOUT,
                speech_detected=False,
                speech_ms=0,
                frame_ms=cfg.frame_ms,
            )

        while len(frames) < max_record_frames:
            frame = read_frame()
            self._validate_frame(frame)
            frames.append(frame)
            partial_frames += 1
            if (
                on_partial is not None
                and speech_started
                and partial_frames >= partial_interval_frames
            ):
                partial_frames = 0
                try:
                    on_partial(
                        b"".join(frames),
                        len(frames) * cfg.frame_ms,
                    )
                except Exception as exc:
                    warnings.warn(
                        f"partial audio callback failed: {type(exc).__name__}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if self.detector.is_speech(frame):
                voiced_frames += 1
                silence_run = 0
            else:
                silence_run += 1

            if silence_run >= end_silence_frames:
                if voiced_frames < min_speech_frames:
                    return RecordingResult(
                        pcm=b"",
                        sample_rate=self.sample_rate,
                        stop_reason=StopReason.TOO_SHORT,
                        speech_detected=False,
                        speech_ms=voiced_frames * cfg.frame_ms,
                        frame_ms=cfg.frame_ms,
                    )
                trim_count = max(0, silence_run - trailing_frames)
                if trim_count:
                    del frames[-trim_count:]
                return RecordingResult(
                    pcm=b"".join(frames),
                    sample_rate=self.sample_rate,
                    stop_reason=StopReason.END_SILENCE,
                    speech_detected=speech_started,
                    speech_ms=voiced_frames * cfg.frame_ms,
                    frame_ms=cfg.frame_ms,
                )

        return RecordingResult(
            pcm=b"".join(frames[:max_record_frames]),
            sample_rate=self.sample_rate,
            stop_reason=StopReason.MAX_DURATION,
            speech_detected=speech_started,
            speech_ms=voiced_frames * cfg.frame_ms,
            frame_ms=cfg.frame_ms,
        )

    def _validate_frame(self, frame: bytes) -> None:
        if len(frame) != self.bytes_per_frame:
            raise ValueError(
                f"expected {self.bytes_per_frame} bytes per audio frame, "
                f"received {len(frame)}"
            )
