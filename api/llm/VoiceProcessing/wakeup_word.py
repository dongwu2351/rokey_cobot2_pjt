from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly


MODEL_PATH = Path(os.getenv(
    "WAKE_WORD_MODEL_PATH",
    str(Path(__file__).with_name("hello_rokey_8332_32.tflite")),
))


class WakeupWord:
    """Incremental openWakeWord detector fed by the pipeline's audio stream."""

    def __init__(
        self,
        buffer_size: int | None = None,
        *,
        model_path: Path = MODEL_PATH,
        threshold: float = 0.3,
        input_rate: int = 16_000,
        prediction_samples: int = 1_280,
        handoff_ms: int = 30,
        model: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_name = self.model_path.stem
        self.threshold = threshold
        self.input_rate = input_rate
        self.prediction_samples = prediction_samples
        if handoff_ms < 0:
            raise ValueError("handoff_ms cannot be negative")
        self.handoff_samples = min(
            prediction_samples,
            round(16_000 * handoff_ms / 1_000),
        )
        self.buffer_size = buffer_size or prediction_samples
        self.model = model
        self.stream = None
        self._samples = np.empty(0, dtype=np.int16)
        self._detection_tail = np.empty(0, dtype=np.int16)
        self._last_sample_rate = input_rate

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"wake-word model not found: {self.model_path}")
        from openwakeword.model import Model

        try:
            self.model = Model(
                wakeword_models=[str(self.model_path)],
                inference_framework="tflite",
            )
        except Exception as exc:
            raise RuntimeError(
                "openWakeWord support assets are missing or invalid. "
                "Run the one-time model download described in VoiceProcessing/README.md."
            ) from exc

    def reset(self) -> None:
        self._samples = np.empty(0, dtype=np.int16)
        self._detection_tail = np.empty(0, dtype=np.int16)
        if self.model is not None and hasattr(self.model, "reset"):
            self.model.reset()

    def process(self, pcm: bytes, *, sample_rate: int | None = None) -> bool:
        self.load()
        rate = sample_rate or self.input_rate
        self._last_sample_rate = rate
        samples = np.frombuffer(pcm, dtype=np.int16)
        if rate != 16_000:
            divisor = int(np.gcd(rate, 16_000))
            samples = resample_poly(
                samples,
                16_000 // divisor,
                rate // divisor,
            ).astype(np.int16)
        self._samples = np.concatenate((self._samples, samples))

        detected = False
        while self._samples.size >= self.prediction_samples:
            frame = self._samples[: self.prediction_samples]
            self._samples = self._samples[self.prediction_samples :]
            outputs = self.model.predict(frame)
            confidence = float(outputs.get(self.model_name, 0.0))
            if confidence >= self.threshold:
                if self.handoff_samples:
                    self._detection_tail = frame[-self.handoff_samples :].copy()
                detected = True
                break
        return detected

    def drain_remainder(self, *, sample_rate: int | None = None) -> bytes:
        """Return the detected frame tail plus audio already read after it.

        Keeping one 30 ms tail protects a command's first syllable when it starts
        inside the 80 ms wake inference frame.  It is shorter than the VAD's
        default pipeline's two-frame onset requirement, so wake-word speech alone
        cannot start a recording.
        """
        samples = np.concatenate((self._detection_tail, self._samples))
        self._detection_tail = np.empty(0, dtype=np.int16)
        self._samples = np.empty(0, dtype=np.int16)
        target_rate = sample_rate or self._last_sample_rate
        if samples.size and target_rate != 16_000:
            divisor = int(np.gcd(target_rate, 16_000))
            samples = resample_poly(
                samples,
                target_rate // divisor,
                16_000 // divisor,
            ).astype(np.int16)
        return samples.tobytes()

    def set_stream(self, stream) -> None:
        """Compatibility adapter for the original polling interface."""
        self.load()
        self.stream = stream

    def is_wakeup(self) -> bool:
        if self.stream is None:
            raise RuntimeError("wake-word stream is not set")
        pcm = self.stream.read(self.buffer_size, exception_on_overflow=False)
        return self.process(pcm, sample_rate=self.input_rate)


if __name__ == "__main__":
    try:
        from .MicController import MicController
    except ImportError:
        from MicController import MicController

    with MicController() as mic:
        wakeup = WakeupWord(input_rate=mic.config.rate)
        print("웨이크워드를 기다립니다...")
        while not wakeup.process(mic.read_frame(), sample_rate=mic.config.rate):
            pass
        print("웨이크워드를 감지했습니다.")
