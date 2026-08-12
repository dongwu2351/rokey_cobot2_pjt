from __future__ import annotations

import io
import time
import unittest
import warnings
import wave
from collections import deque
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from VoiceProcessing.MicController import MicConfig, MicController
from VoiceProcessing.STT import STT, STTConfig
from VoiceProcessing.command_models import (
    Ambiguity,
    CommandResult,
    Decision,
    ExecutableAction,
    Intent,
)
from VoiceProcessing.command_router import CommandRouter
from VoiceProcessing.vad import (
    AdaptiveEnergyVAD,
    EndpointRecorder,
    RecordingResult,
    StopReason,
    VADConfig,
)
from VoiceProcessing.voice_pipeline import PipelineState, VoiceCommandPipeline
from VoiceProcessing.wakeup_word import WakeupWord


SAMPLE_RATE = 16_000
FRAME_MS = 10


def grounded_context(
    object_name: str = "hammer",
    *,
    timestamp_ms: int | None = None,
) -> dict[str, Any]:
    return {
        "robot_state": "ready",
        "visible_objects": [
            {
                "id": f"{object_name}-1",
                "canonical_name": object_name,
                "snapshot_revision": "test-camera-1",
            }
        ],
        "snapshot_revision": "test-camera-1",
        "snapshot_timestamp_ms": (
            timestamp_ms if timestamp_ms is not None else round(time.time() * 1_000)
        ),
    }


def pcm_frame(
    value: int = 0,
    *,
    sample_rate: int = SAMPLE_RATE,
    frame_ms: int = FRAME_MS,
) -> bytes:
    sample_count = sample_rate * frame_ms // 1_000
    return np.full(sample_count, value, dtype=np.int16).tobytes()


class FrameReader:
    def __init__(self, frames: list[bytes]) -> None:
        self.frames = deque(frames)
        self.calls = 0

    def __call__(self) -> bytes:
        self.calls += 1
        if not self.frames:
            raise AssertionError("the recorder read more frames than the test supplied")
        return self.frames.popleft()


class AmplitudeDetector:
    """Deterministic VAD: zero is silence and any non-zero frame is speech."""

    def __init__(self) -> None:
        self.reset_calls = 0
        self.calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def is_speech(self, pcm: bytes) -> bool:
        self.calls += 1
        return bool(np.any(np.frombuffer(pcm, dtype=np.int16)))


def recorder_config(**overrides: Any) -> VADConfig:
    values: dict[str, Any] = {
        "frame_ms": FRAME_MS,
        "start_timeout_ms": 50,
        "max_record_ms": 200,
        "onset_frames": 2,
        "end_silence_ms": 20,
        "pre_roll_ms": 20,
        "trailing_silence_ms": 10,
        "min_speech_ms": 20,
    }
    values.update(overrides)
    return VADConfig(**values)


class EndpointRecorderTests(unittest.TestCase):
    def test_start_timeout_returns_no_audio_and_does_not_claim_speech(self) -> None:
        detector = AmplitudeDetector()
        recorder = EndpointRecorder(
            detector,
            sample_rate=SAMPLE_RATE,
            config=recorder_config(start_timeout_ms=30),
        )
        reader = FrameReader([pcm_frame(), pcm_frame(), pcm_frame()])

        result = recorder.record(reader)

        self.assertEqual(result.stop_reason, StopReason.START_TIMEOUT)
        self.assertFalse(result.speech_detected)
        self.assertEqual(result.pcm, b"")
        self.assertEqual(result.duration_ms, 0)
        self.assertEqual(result.speech_ms, 0)
        self.assertEqual(reader.calls, 3)
        self.assertEqual(detector.reset_calls, 1)

    def test_end_silence_keeps_short_pause_and_only_requested_trailing_audio(self) -> None:
        detector = AmplitudeDetector()
        recorder = EndpointRecorder(
            detector,
            sample_rate=SAMPLE_RATE,
            config=recorder_config(),
        )
        silence = pcm_frame()
        speech = pcm_frame(2_000)
        reader = FrameReader(
            [
                silence,
                speech,
                speech,  # onset after two consecutive voiced frames
                speech,
                speech,
                silence,  # 10 ms pause: shorter than the endpoint threshold
                speech,
                silence,
                silence,  # 20 ms silence: endpoint
            ]
        )

        result = recorder.record(reader)

        self.assertEqual(result.stop_reason, StopReason.END_SILENCE)
        self.assertTrue(result.speech_detected)
        self.assertEqual(result.speech_ms, 50)
        self.assertEqual(result.duration_ms, 70)
        self.assertEqual(len(result.pcm), 7 * len(speech))
        self.assertEqual(result.pcm[-len(silence) :], silence)
        self.assertEqual(reader.calls, 9)

    def test_max_duration_caps_continuous_speech(self) -> None:
        detector = AmplitudeDetector()
        recorder = EndpointRecorder(
            detector,
            sample_rate=SAMPLE_RATE,
            config=recorder_config(
                max_record_ms=50,
                onset_frames=1,
                pre_roll_ms=10,
                min_speech_ms=10,
            ),
        )
        speech = pcm_frame(1_500)
        reader = FrameReader([speech] * 8)

        result = recorder.record(reader)

        self.assertEqual(result.stop_reason, StopReason.MAX_DURATION)
        self.assertTrue(result.speech_detected)
        self.assertEqual(result.duration_ms, 50)
        self.assertEqual(result.speech_ms, 50)
        self.assertEqual(len(result.pcm), 5 * len(speech))
        self.assertEqual(reader.calls, 5)

    def test_too_short_false_onset_ends_without_waiting_for_max_duration(self) -> None:
        detector = AmplitudeDetector()
        recorder = EndpointRecorder(
            detector,
            sample_rate=SAMPLE_RATE,
            config=recorder_config(min_speech_ms=40),
        )
        speech = pcm_frame(1_500)
        silence = pcm_frame()
        reader = FrameReader([speech, speech, silence, silence])

        result = recorder.record(reader)

        self.assertEqual(result.stop_reason, StopReason.TOO_SHORT)
        self.assertFalse(result.speech_detected)
        self.assertEqual(result.pcm, b"")
        self.assertEqual(reader.calls, 4)


class AdaptiveEnergyVADTests(unittest.TestCase):
    def test_moderate_stationary_noise_does_not_remain_speech_forever(self) -> None:
        detector = AdaptiveEnergyVAD()
        factory_noise = pcm_frame(400)

        decisions = [detector.is_speech(factory_noise) for _ in range(20)]

        self.assertIn(True, decisions)
        self.assertFalse(decisions[-1])
        self.assertTrue(detector.is_speech(pcm_frame(5_000)))


class WavEncodingTests(unittest.TestCase):
    def test_recording_result_serializes_pcm_with_expected_wav_header(self) -> None:
        pcm = pcm_frame(1_234) + pcm_frame(-1_234)
        result = RecordingResult(
            pcm=pcm,
            sample_rate=SAMPLE_RATE,
            stop_reason=StopReason.END_SILENCE,
            speech_detected=True,
            speech_ms=20,
            frame_ms=FRAME_MS,
        )

        wav_bytes = result.to_wav_bytes()

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), SAMPLE_RATE)
            self.assertEqual(wav_file.getnframes(), SAMPLE_RATE * 20 // 1_000)
            self.assertEqual(wav_file.readframes(wav_file.getnframes()), pcm)


class STTTests(unittest.TestCase):
    def test_transcribe_sends_in_memory_wav_with_korean_context(self) -> None:
        class FakeTranscriptions:
            def __init__(self) -> None:
                self.request = None

            def create(self, **request):
                self.request = request
                return SimpleNamespace(text=" 해머 가져와 ")

        transcriptions = FakeTranscriptions()
        client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=transcriptions)
        )
        stt = STT(
            config=STTConfig(keywords=("해머",), prompt="로봇 명령"),
            client=client,
        )

        text = stt.transcribe(b"RIFF-test-data")

        self.assertEqual(text, "해머 가져와")
        request = transcriptions.request
        self.assertEqual(request["model"], "gpt-transcribe")
        self.assertEqual(request["file"][0], "command.wav")
        self.assertEqual(request["file"][1], b"RIFF-test-data")
        self.assertEqual(request["extra_body"]["languages"], ["ko"])
        self.assertEqual(request["extra_body"]["keywords"], ["해머"])


class FakeWakeModel:
    def __init__(self, model_name: str, scores: list[float]) -> None:
        self.model_name = model_name
        self.scores = deque(scores)
        self.frames: list[np.ndarray] = []
        self.reset_calls = 0

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        self.frames.append(frame.copy())
        if not self.scores:
            raise AssertionError("wake model received an unexpected prediction frame")
        return {self.model_name: self.scores.popleft()}

    def reset(self) -> None:
        self.reset_calls += 1


class WakeupWordTests(unittest.TestCase):
    def test_process_accumulates_partial_input_into_exact_1280_sample_frames(self) -> None:
        model_name = "hello_rokey_8332_32"
        model = FakeWakeModel(model_name, [0.1, 0.8])
        detector = WakeupWord(
            model=model,
            prediction_samples=1_280,
            threshold=0.3,
            input_rate=SAMPLE_RATE,
        )
        audio = np.arange(2_560, dtype=np.int16)
        chunk_sizes = [480, 480, 640, 960]
        results: list[bool] = []
        offset = 0

        for chunk_size in chunk_sizes:
            chunk = audio[offset : offset + chunk_size]
            results.append(detector.process(chunk.tobytes(), sample_rate=SAMPLE_RATE))
            offset += chunk_size

        self.assertEqual(results, [False, False, False, True])
        self.assertEqual(len(model.frames), 2)
        np.testing.assert_array_equal(model.frames[0], audio[:1_280])
        np.testing.assert_array_equal(model.frames[1], audio[1_280:])

    def test_detected_frame_remainder_can_be_handed_to_vad(self) -> None:
        model_name = "hello_rokey_8332_32"
        model = FakeWakeModel(model_name, [0.8])
        detector = WakeupWord(model=model, threshold=0.3, input_rate=SAMPLE_RATE)
        audio = np.arange(1_440, dtype=np.int16)

        self.assertTrue(detector.process(audio.tobytes(), sample_rate=SAMPLE_RATE))
        remainder = np.frombuffer(
            detector.drain_remainder(sample_rate=SAMPLE_RATE),
            dtype=np.int16,
        )

        # Default handoff preserves the final 30 ms (480 samples) of the
        # detected wake frame as well as the 160 samples read after it.
        np.testing.assert_array_equal(remainder, audio[800:])
        self.assertEqual(detector.drain_remainder(), b"")


class FakePortAudioStream:
    def __init__(self, frame: bytes) -> None:
        self.frame = frame
        self.read_calls: list[tuple[int, bool]] = []
        self.active = True
        self.stop_calls = 0
        self.close_calls = 0

    def read(self, samples: int, *, exception_on_overflow: bool) -> bytes:
        self.read_calls.append((samples, exception_on_overflow))
        return self.frame

    def is_active(self) -> bool:
        return self.active

    def stop_stream(self) -> None:
        self.stop_calls += 1
        self.active = False

    def close(self) -> None:
        self.close_calls += 1


class FakePortAudio:
    def __init__(self, stream: FakePortAudioStream) -> None:
        self.stream = stream
        self.open_calls: list[dict[str, Any]] = []
        self.terminate_calls = 0

    def get_sample_size(self, _audio_format: int) -> int:
        return 2

    def open(self, **kwargs: Any) -> FakePortAudioStream:
        self.open_calls.append(kwargs)
        return self.stream

    def terminate(self) -> None:
        self.terminate_calls += 1


class MicControllerTests(unittest.TestCase):
    def test_open_is_idempotent_and_reads_use_the_same_stream(self) -> None:
        frame = pcm_frame()
        stream = FakePortAudioStream(frame)
        audio = FakePortAudio(stream)
        factory_calls = 0

        def audio_factory() -> FakePortAudio:
            nonlocal factory_calls
            factory_calls += 1
            return audio

        mic = MicController(
            MicConfig(rate=SAMPLE_RATE, frame_ms=FRAME_MS, device_index=7),
            audio_factory=audio_factory,
        )

        mic.open_stream()
        mic.open_stream()
        first = mic.read_frame()
        second = mic.read_frame()
        mic.close_stream()

        self.assertEqual(first, frame)
        self.assertEqual(second, frame)
        self.assertEqual(factory_calls, 1)
        self.assertEqual(len(audio.open_calls), 1)
        self.assertEqual(audio.open_calls[0]["frames_per_buffer"], 160)
        self.assertEqual(audio.open_calls[0]["input_device_index"], 7)
        self.assertEqual(stream.read_calls, [(160, False), (160, False)])
        self.assertEqual(stream.stop_calls, 1)
        self.assertEqual(stream.close_calls, 1)
        self.assertEqual(audio.terminate_calls, 1)
        self.assertFalse(mic.is_open)


class FakeMicrophone:
    def __init__(self, frames: list[bytes]) -> None:
        self.config = SimpleNamespace(rate=SAMPLE_RATE)
        self.frames = deque(frames)
        self._is_open = False
        self.open_calls = 0
        self.close_calls = 0
        self.read_calls = 0

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open_stream(self) -> None:
        if not self._is_open:
            self.open_calls += 1
            self._is_open = True

    def close_stream(self) -> None:
        if self._is_open:
            self.close_calls += 1
            self._is_open = False

    def read_frame(self) -> bytes:
        if not self._is_open:
            raise AssertionError("pipeline tried to read a closed microphone")
        self.read_calls += 1
        if not self.frames:
            raise AssertionError("pipeline read more microphone frames than supplied")
        return self.frames.popleft()


class FakeWakeDetector:
    def __init__(self, detect_on_call: int) -> None:
        self.detect_on_call = detect_on_call
        self.frames: list[bytes] = []
        self.sample_rates: list[int] = []
        self.reset_calls = 0

    def process(self, frame: bytes, *, sample_rate: int) -> bool:
        self.frames.append(frame)
        self.sample_rates.append(sample_rate)
        return len(self.frames) >= self.detect_on_call

    def reset(self) -> None:
        self.reset_calls += 1


class FakeTranscriber:
    def __init__(
        self,
        text: str,
        *,
        before_transcribe: Callable[[], None] | None = None,
    ) -> None:
        self.text = text
        self.before_transcribe = before_transcribe
        self.wav_calls: list[bytes] = []

    def transcribe(self, wav_data: bytes) -> str:
        if self.before_transcribe is not None:
            self.before_transcribe()
        self.wav_calls.append(wav_data)
        return self.text


class FakeRouter:
    def __init__(
        self,
        *,
        before_parse: Callable[[], None] | None = None,
    ) -> None:
        self.before_parse = before_parse
        self.command = self._command(
            revision="test-camera-1",
            timestamp_ms=round(time.time() * 1_000),
        )
        self.calls: list[tuple[str, Any]] = []

    @staticmethod
    def _command(*, revision: str, timestamp_ms: int) -> CommandResult:
        return CommandResult(
            decision=Decision.READY,
            actions=[
                ExecutableAction(
                    intent=Intent.FETCH,
                    object="hammer",
                    destination=None,
                    object_query="해머",
                    resolved_object_id="hammer-1",
                )
            ],
            ambiguity=Ambiguity.NONE,
            clarification_question=None,
            route="FAST_RULE",
            raw_utterance="해머 가져와",
            snapshot_revision=revision,
            snapshot_timestamp_ms=timestamp_ms,
        )

    def parse_command(self, transcript: str, context: Any) -> object:
        if self.before_parse is not None:
            self.before_parse()
        self.calls.append((transcript, context))
        if isinstance(context, dict):
            self.command = self._command(
                revision=context["snapshot_revision"],
                timestamp_ms=context["snapshot_timestamp_ms"],
            )
        return self.command


class VoiceCommandPipelineTests(unittest.TestCase):
    def test_wake_and_record_share_one_capture_closed_before_transcribe(self) -> None:
        wake_a = pcm_frame(101)
        wake_b = pcm_frame(102)
        speech = pcm_frame(2_000)
        silence = pcm_frame()
        microphone = FakeMicrophone(
            [wake_a, wake_b, speech, speech, silence, silence]
        )
        wake_detector = FakeWakeDetector(detect_on_call=2)
        recorder = EndpointRecorder(
            AmplitudeDetector(),
            sample_rate=SAMPLE_RATE,
            config=recorder_config(
                start_timeout_ms=20,
                max_record_ms=100,
                onset_frames=1,
                pre_roll_ms=10,
                min_speech_ms=10,
            ),
        )
        transcriber = FakeTranscriber(
            "해머 가져와",
            before_transcribe=lambda: self.assertFalse(microphone.is_open),
        )
        router = FakeRouter(
            before_parse=lambda: self.assertFalse(microphone.is_open)
        )
        context = grounded_context()
        pipeline = VoiceCommandPipeline(
            microphone=microphone,
            recorder=recorder,
            transcriber=transcriber,
            router=router,
            wake_detector=wake_detector,
        )

        result = pipeline.run_once(
            wait_for_wake=True,
            context_provider=lambda: context,
        )

        self.assertEqual(result.state, PipelineState.COMMAND_READY)
        self.assertTrue(result.executable)
        self.assertEqual(result.transcript, "해머 가져와")
        self.assertIs(result.command, router.command)
        self.assertEqual(microphone.open_calls, 1)
        self.assertEqual(microphone.close_calls, 1)
        self.assertEqual(microphone.read_calls, 6)
        self.assertEqual(wake_detector.frames, [wake_a, wake_b])
        self.assertEqual(wake_detector.sample_rates, [SAMPLE_RATE, SAMPLE_RATE])
        self.assertEqual(wake_detector.reset_calls, 1)
        self.assertEqual(len(transcriber.wav_calls), 1)
        self.assertEqual(router.calls, [("해머 가져와", context)])
        self.assertIsNotNone(result.recording)
        self.assertEqual(result.recording.stop_reason, StopReason.END_SILENCE)

        with wave.open(io.BytesIO(transcriber.wav_calls[0]), "rb") as wav_file:
            self.assertEqual(wav_file.getframerate(), SAMPLE_RATE)
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getnframes(), SAMPLE_RATE * 30 // 1_000)

    def test_context_provider_is_called_after_transcription(self) -> None:
        speech = pcm_frame(2_000)
        silence = pcm_frame()
        microphone = FakeMicrophone([speech, speech, silence, silence])
        recorder = EndpointRecorder(
            AmplitudeDetector(),
            sample_rate=SAMPLE_RATE,
            config=recorder_config(onset_frames=1, pre_roll_ms=10),
        )
        transcriber = FakeTranscriber("해머 가져와")
        router = FakeRouter()
        context = grounded_context()
        pipeline = VoiceCommandPipeline(
            microphone=microphone,
            recorder=recorder,
            transcriber=transcriber,
            router=router,
        )

        def current_context():
            self.assertEqual(len(transcriber.wav_calls), 1)
            self.assertFalse(microphone.is_open)
            return context

        result = pipeline.run_once(
            wait_for_wake=False,
            context_provider=current_context,
        )

        self.assertEqual(result.state, PipelineState.COMMAND_READY)
        self.assertEqual(router.calls[0][1], context)

    def test_context_expiry_is_one_way_across_suspend_and_wall_rollback(self) -> None:
        wall_now = [100.0]
        monotonic_now = [50.0]
        speech = pcm_frame(2_000)
        silence = pcm_frame()
        microphone = FakeMicrophone([speech, speech, silence, silence])
        pipeline = VoiceCommandPipeline(
            microphone=microphone,
            recorder=EndpointRecorder(
                AmplitudeDetector(),
                sample_rate=SAMPLE_RATE,
                config=recorder_config(onset_frames=1, pre_roll_ms=10),
            ),
            transcriber=FakeTranscriber("해머 가져와"),
            router=FakeRouter(),
            wall_clock=lambda: wall_now[0],
            expiry_clock=lambda: monotonic_now[0],
            max_context_age_ms=1_000,
        )

        result = pipeline.run_once(
            wait_for_wake=False,
            context_provider=lambda: grounded_context(timestamp_ms=99_500),
        )

        self.assertTrue(result.safety_context_fresh)
        self.assertTrue(result.executable)
        self.assertEqual(result.context_valid_until_ms, 100_500)

        # Simulate a suspend/VM pause: the injected expiry clock did not move,
        # but wall time passed the TTL. The wall gate expires and latches it.
        wall_now[0] = 104.0
        self.assertFalse(result.safety_context_fresh)
        self.assertFalse(result.executable)

        # A later wall-clock rollback cannot resurrect the expired command.
        wall_now[0] = 99.0
        self.assertFalse(result.safety_context_fresh)
        self.assertFalse(result.executable)

        # A normally advancing suspend-aware clock is an independent gate.
        wall_now[0] = 100.0
        monotonic_now[0] = 60.0
        second_pipeline = VoiceCommandPipeline(
            microphone=FakeMicrophone([speech, speech, silence, silence]),
            recorder=EndpointRecorder(
                AmplitudeDetector(),
                sample_rate=SAMPLE_RATE,
                config=recorder_config(onset_frames=1, pre_roll_ms=10),
            ),
            transcriber=FakeTranscriber("해머 가져와"),
            router=FakeRouter(),
            wall_clock=lambda: wall_now[0],
            expiry_clock=lambda: monotonic_now[0],
            max_context_age_ms=1_000,
        )
        second_result = second_pipeline.run_once(
            wait_for_wake=False,
            context_provider=lambda: grounded_context(timestamp_ms=99_500),
        )
        self.assertTrue(second_result.executable)

        monotonic_now[0] = 60.501
        self.assertFalse(second_result.safety_context_fresh)
        self.assertFalse(second_result.executable)

    def test_preopened_run_once_is_rejected_before_audio_or_cloud_work(self) -> None:
        microphone = FakeMicrophone([pcm_frame(2_000)])
        microphone.open_stream()
        transcriber = FakeTranscriber("해머 가져와")
        router = FakeRouter()
        pipeline = VoiceCommandPipeline(
            microphone=microphone,
            recorder=EndpointRecorder(
                AmplitudeDetector(),
                sample_rate=SAMPLE_RATE,
                config=recorder_config(onset_frames=1, pre_roll_ms=10),
            ),
            transcriber=transcriber,
            router=router,
        )

        result = pipeline.run_once(wait_for_wake=False)

        self.assertEqual(result.state, PipelineState.ERROR)
        self.assertEqual(result.error, "RuntimeError")
        self.assertTrue(microphone.is_open)
        self.assertEqual(microphone.read_calls, 0)
        self.assertEqual(transcriber.wav_calls, [])
        self.assertEqual(router.calls, [])
        microphone.close_stream()

    def test_nonconforming_router_result_fails_closed_and_serializes(self) -> None:
        class InvalidRouter:
            def parse_command(self, transcript, context):
                return SimpleNamespace(decision=Decision.READY, actions=[])

        speech = pcm_frame(2_000)
        silence = pcm_frame()
        pipeline = VoiceCommandPipeline(
            microphone=FakeMicrophone([speech, speech, silence, silence]),
            recorder=EndpointRecorder(
                AmplitudeDetector(),
                sample_rate=SAMPLE_RATE,
                config=recorder_config(onset_frames=1, pre_roll_ms=10),
            ),
            transcriber=FakeTranscriber("해머 가져와"),
            router=InvalidRouter(),
        )

        result = pipeline.run_once(
            wait_for_wake=False,
            context_provider=grounded_context,
        )

        self.assertEqual(result.state, PipelineState.ERROR)
        self.assertEqual(result.error, "TypeError")
        self.assertIsNone(result.command)
        self.assertFalse(result.to_dict()["executable"])

    def test_real_router_promotes_only_grounded_motion_to_executable(self) -> None:
        speech = pcm_frame(2_000)
        silence = pcm_frame()
        microphone = FakeMicrophone([speech, speech, silence, silence])
        pipeline = VoiceCommandPipeline(
            microphone=microphone,
            recorder=EndpointRecorder(
                AmplitudeDetector(),
                sample_rate=SAMPLE_RATE,
                config=recorder_config(onset_frames=1, pre_roll_ms=10),
            ),
            transcriber=FakeTranscriber("해머 가져와"),
            router=CommandRouter(),
        )

        result = pipeline.run_once(
            wait_for_wake=False,
            context_provider=grounded_context,
        )

        self.assertEqual(result.state, PipelineState.COMMAND_READY)
        self.assertTrue(result.executable)
        self.assertEqual(result.command.actions[0].resolved_object_id, "hammer-1")
        self.assertEqual(result.command.snapshot_revision, "test-camera-1")

        redacted = result.to_dict(include_transcript=False)
        self.assertIsNone(redacted["transcript"])
        self.assertEqual(redacted["command"]["raw_utterance"], "[REDACTED]")

    def test_max_duration_never_reaches_stt_or_router(self) -> None:
        speech = pcm_frame(2_000)
        microphone = FakeMicrophone([speech] * 8)
        recorder = EndpointRecorder(
            AmplitudeDetector(),
            sample_rate=SAMPLE_RATE,
            config=recorder_config(
                max_record_ms=50,
                onset_frames=1,
                pre_roll_ms=10,
                min_speech_ms=10,
            ),
        )
        transcriber = FakeTranscriber("해머 가져와")
        router = FakeRouter()
        pipeline = VoiceCommandPipeline(
            microphone=microphone,
            recorder=recorder,
            transcriber=transcriber,
            router=router,
        )

        result = pipeline.run_once(wait_for_wake=False)

        self.assertEqual(result.state, PipelineState.TRUNCATED)
        self.assertFalse(result.executable)
        self.assertEqual(transcriber.wav_calls, [])
        self.assertEqual(router.calls, [])

    def test_motion_is_not_executable_without_fresh_safety_context(self) -> None:
        speech = pcm_frame(2_000)
        silence = pcm_frame()
        microphone = FakeMicrophone([speech, speech, silence, silence])
        recorder = EndpointRecorder(
            AmplitudeDetector(),
            sample_rate=SAMPLE_RATE,
            config=recorder_config(onset_frames=1, pre_roll_ms=10),
        )
        pipeline = VoiceCommandPipeline(
            microphone=microphone,
            recorder=recorder,
            transcriber=FakeTranscriber("해머 가져와"),
            router=FakeRouter(),
        )

        result = pipeline.run_once(wait_for_wake=False)

        self.assertEqual(result.command.decision, Decision.READY)
        self.assertEqual(result.state, PipelineState.COMMAND_NOT_READY)
        self.assertFalse(result.executable)
        self.assertFalse(result.safety_context_fresh)

    def test_wake_cleanup_failure_does_not_leak_microphone(self) -> None:
        class BrokenResetWake(FakeWakeDetector):
            def reset(self) -> None:
                raise RuntimeError("reset failed")

        wake = pcm_frame(100)
        speech = pcm_frame(2_000)
        silence = pcm_frame()
        microphone = FakeMicrophone([wake, speech, speech, silence, silence])
        recorder = EndpointRecorder(
            AmplitudeDetector(),
            sample_rate=SAMPLE_RATE,
            config=recorder_config(onset_frames=1, pre_roll_ms=10),
        )
        pipeline = VoiceCommandPipeline(
            microphone=microphone,
            recorder=recorder,
            transcriber=FakeTranscriber("해머 가져와"),
            router=FakeRouter(),
            wake_detector=BrokenResetWake(detect_on_call=1),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = pipeline.run_once(
                wait_for_wake=True,
                context_provider=grounded_context,
            )

        self.assertEqual(result.state, PipelineState.COMMAND_READY)
        self.assertFalse(microphone.is_open)
        self.assertEqual(microphone.close_calls, 1)
        self.assertEqual(len(caught), 1)

    def test_close_failure_blocks_stt_and_is_retried_during_cleanup(self) -> None:
        class FailOnceCloseMicrophone(FakeMicrophone):
            def __init__(self, frames: list[bytes]) -> None:
                super().__init__(frames)
                self.close_attempts = 0

            def close_stream(self) -> None:
                self.close_attempts += 1
                if self.close_attempts == 1:
                    raise RuntimeError("close failed")
                super().close_stream()

        speech = pcm_frame(2_000)
        silence = pcm_frame()
        microphone = FailOnceCloseMicrophone(
            [speech, speech, silence, silence]
        )
        transcriber = FakeTranscriber("해머 가져와")
        router = FakeRouter()
        pipeline = VoiceCommandPipeline(
            microphone=microphone,
            recorder=EndpointRecorder(
                AmplitudeDetector(),
                sample_rate=SAMPLE_RATE,
                config=recorder_config(onset_frames=1, pre_roll_ms=10),
            ),
            transcriber=transcriber,
            router=router,
        )

        result = pipeline.run_once(wait_for_wake=False)

        self.assertEqual(result.state, PipelineState.ERROR)
        self.assertEqual(result.error, "RuntimeError")
        self.assertEqual(microphone.close_attempts, 2)
        self.assertEqual(microphone.close_calls, 1)
        self.assertFalse(microphone.is_open)
        self.assertEqual(transcriber.wav_calls, [])
        self.assertEqual(router.calls, [])

    def test_run_forever_opens_fresh_half_duplex_session_each_iteration(self) -> None:
        class StopAfterTwoCommands(Exception):
            pass

        class SessionMicrophone(FakeMicrophone):
            def __init__(self, sessions: list[list[bytes]]) -> None:
                super().__init__([])
                self.sessions = deque(deque(session) for session in sessions)

            def open_stream(self) -> None:
                if self._is_open:
                    raise AssertionError("continuous mode reopened a live stream")
                if not self.sessions:
                    raise AssertionError("continuous mode opened an unexpected session")
                self.frames = self.sessions.popleft()
                self.open_calls += 1
                self._is_open = True

            def close_stream(self) -> None:
                if self._is_open and self.frames:
                    raise AssertionError(
                        "capture session closed before its scripted frames were consumed"
                    )
                super().close_stream()

        class SequenceTranscriber(FakeTranscriber):
            def __init__(self, microphone: SessionMicrophone) -> None:
                super().__init__("", before_transcribe=self._assert_closed)
                self.microphone = microphone
                self.texts = deque(["해머 가져와", "렌치 가져와"])

            def _assert_closed(self) -> None:
                if self.microphone.is_open:
                    raise AssertionError("STT started while capture was still open")

            def transcribe(self, wav_data: bytes) -> str:
                super().transcribe(wav_data)
                return self.texts.popleft()

        wake_one = pcm_frame(101)
        wake_two = pcm_frame(202)
        speech = pcm_frame(2_000)
        silence = pcm_frame()
        microphone = SessionMicrophone(
            [
                [wake_one, speech, speech, silence, silence],
                [wake_two, speech, speech, silence, silence],
            ]
        )
        wake_detector = FakeWakeDetector(detect_on_call=1)
        transcriber = SequenceTranscriber(microphone)
        pipeline = VoiceCommandPipeline(
            microphone=microphone,
            recorder=EndpointRecorder(
                AmplitudeDetector(),
                sample_rate=SAMPLE_RATE,
                config=recorder_config(onset_frames=1, pre_roll_ms=10),
            ),
            transcriber=transcriber,
            router=FakeRouter(
                before_parse=lambda: self.assertFalse(microphone.is_open)
            ),
            wake_detector=wake_detector,
        )
        results = []

        def callback(result) -> None:
            self.assertFalse(microphone.is_open)
            results.append(result)
            if len(results) == 2:
                raise StopAfterTwoCommands

        with self.assertRaises(StopAfterTwoCommands):
            pipeline.run_forever(
                callback,
                context_provider=grounded_context,
            )

        self.assertEqual(microphone.open_calls, 2)
        self.assertEqual(microphone.close_calls, 2)
        self.assertFalse(microphone.is_open)
        self.assertEqual(len(microphone.sessions), 0)
        self.assertEqual(wake_detector.frames, [wake_one, wake_two])
        self.assertEqual(wake_detector.reset_calls, 2)
        self.assertEqual(
            [result.transcript for result in results],
            ["해머 가져와", "렌치 가져와"],
        )

    def test_run_forever_stops_before_callback_if_stream_cannot_close(self) -> None:
        class NeverCloseMicrophone(FakeMicrophone):
            def __init__(self, frames: list[bytes]) -> None:
                super().__init__(frames)
                self.close_attempts = 0

            def close_stream(self) -> None:
                self.close_attempts += 1
                raise RuntimeError("close failed permanently")

        wake = pcm_frame(100)
        speech = pcm_frame(2_000)
        silence = pcm_frame()
        microphone = NeverCloseMicrophone(
            [wake, speech, speech, silence, silence]
        )
        transcriber = FakeTranscriber("해머 가져와")
        callback_calls = []
        pipeline = VoiceCommandPipeline(
            microphone=microphone,
            recorder=EndpointRecorder(
                AmplitudeDetector(),
                sample_rate=SAMPLE_RATE,
                config=recorder_config(onset_frames=1, pre_roll_ms=10),
            ),
            transcriber=transcriber,
            router=FakeRouter(),
            wake_detector=FakeWakeDetector(detect_on_call=1),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaisesRegex(
                RuntimeError,
                "microphone remained open",
            ):
                pipeline.run_forever(callback_calls.append)

        self.assertEqual(callback_calls, [])
        self.assertEqual(transcriber.wav_calls, [])
        self.assertEqual(microphone.close_attempts, 3)
        self.assertTrue(microphone.is_open)
        self.assertEqual(len(caught), 2)


if __name__ == "__main__":
    unittest.main()
