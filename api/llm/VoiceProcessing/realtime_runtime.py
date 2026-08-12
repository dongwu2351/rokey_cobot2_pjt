from __future__ import annotations

import json
import multiprocessing as mp
import queue
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from multiprocessing import shared_memory
from typing import Any, Callable

import cv2
import numpy as np

from .assistive_models import AssistiveResult, AssistiveState, TrackedObject
from .assistive_processor import AssistiveCommandProcessor


@dataclass(frozen=True)
class FrameSnapshot:
    sequence: int
    captured_at: float
    frame: np.ndarray


class LatestFrameCamera:
    """Continuously drain a camera into a one-frame, low-latency buffer."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self.capture = capture
        self._condition = threading.Condition()
        self._latest: FrameSnapshot | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fps = 0.0
        self._frame_times: deque[float] = deque(maxlen=120)

    @property
    def fps(self) -> float:
        with self._condition:
            return self._fps

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._capture_loop, name="assistive-camera", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def latest(self, *, wait_seconds: float = 2.0, copy: bool = True) -> FrameSnapshot:
        deadline = time.monotonic() + wait_seconds
        with self._condition:
            while self._latest is None and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("timed out waiting for a webcam frame")
                self._condition.wait(remaining)
            if self._latest is None:
                raise RuntimeError("webcam stopped before producing a frame")
            snapshot = self._latest
            return FrameSnapshot(
                sequence=snapshot.sequence,
                captured_at=snapshot.captured_at,
                frame=snapshot.frame.copy() if copy else snapshot.frame,
            )

    def _capture_loop(self) -> None:
        sequence = 0
        while not self._stop.is_set():
            ok, frame = self.capture.read()
            captured_at = time.monotonic()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            sequence += 1
            with self._condition:
                self._frame_times.append(captured_at)
                if len(self._frame_times) >= 2:
                    elapsed = self._frame_times[-1] - self._frame_times[0]
                    if elapsed > 0:
                        self._fps = (len(self._frame_times) - 1) / elapsed
                self._latest = FrameSnapshot(sequence, captured_at, frame)
                self._condition.notify_all()


def _camera_process_main(
    *,
    shared_name: str,
    shape: tuple[int, int, int],
    camera_index: int,
    requested_fps: float,
    sequence: Any,
    captured_at: Any,
    measured_fps: Any,
    lock: Any,
    ready: Any,
    stop: Any,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    memory = shared_memory.SharedMemory(name=shared_name)
    shared_frame = np.ndarray(shape, dtype=np.uint8, buffer=memory.buf)
    capture = cv2.VideoCapture(camera_index)
    try:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, shape[1])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, shape[0])
        capture.set(cv2.CAP_PROP_FPS, requested_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            ready.set()
            return
        frame_times: deque[float] = deque(maxlen=120)
        local_sequence = 0
        while not stop.is_set():
            ok, frame = capture.read()
            now = time.monotonic()
            if not ok or frame is None or frame.shape != shape:
                time.sleep(0.005)
                continue
            local_sequence += 1
            frame_times.append(now)
            fps = 0.0
            if len(frame_times) >= 2:
                elapsed = frame_times[-1] - frame_times[0]
                if elapsed > 0:
                    fps = (len(frame_times) - 1) / elapsed
            with lock:
                np.copyto(shared_frame, frame)
                sequence.value = local_sequence
                captured_at.value = now
                measured_fps.value = fps
            ready.set()
    finally:
        capture.release()
        memory.close()


class ProcessFrameCamera:
    """GIL-isolated camera capture backed by a single shared-memory frame."""

    def __init__(
        self,
        *,
        camera_index: int,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        self.shape = (height, width, 3)
        self.camera_index = camera_index
        self.requested_fps = fps
        self._context = mp.get_context("fork")
        self._memory = shared_memory.SharedMemory(
            create=True, size=int(np.prod(self.shape))
        )
        self._sequence = self._context.Value("Q", 0)
        self._captured_at = self._context.Value("d", 0.0)
        self._fps = self._context.Value("d", 0.0)
        self._lock = self._context.Lock()
        self._ready = self._context.Event()
        self._stop = self._context.Event()
        self._process: mp.Process | None = None
        self._closed = False

    @property
    def fps(self) -> float:
        with self._lock:
            return float(self._fps.value)

    def start(self) -> None:
        if self._process is not None:
            return
        self._process = self._context.Process(
            target=_camera_process_main,
            kwargs={
                "shared_name": self._memory.name,
                "shape": self.shape,
                "camera_index": self.camera_index,
                "requested_fps": self.requested_fps,
                "sequence": self._sequence,
                "captured_at": self._captured_at,
                "measured_fps": self._fps,
                "lock": self._lock,
                "ready": self._ready,
                "stop": self._stop,
            },
            name="assistive-camera-process",
            daemon=True,
        )
        self._process.start()

    def latest(self, *, wait_seconds: float = 2.0, copy: bool = True) -> FrameSnapshot:
        if not self._ready.wait(wait_seconds):
            raise RuntimeError("timed out waiting for camera process")
        if self._process is not None and not self._process.is_alive():
            raise RuntimeError("camera process exited before producing a frame")
        with self._lock:
            frame = np.ndarray(
                self.shape, dtype=np.uint8, buffer=self._memory.buf
            ).copy()
            return FrameSnapshot(
                sequence=int(self._sequence.value),
                captured_at=float(self._captured_at.value),
                frame=frame,
            )

    def stop(self) -> None:
        if self._closed:
            return
        self._stop.set()
        if self._process is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        self._memory.close()
        self._memory.unlink()
        self._closed = True


@dataclass
class OverlayObject:
    id: str
    label: str
    score: float
    bbox: np.ndarray
    points: np.ndarray | None = None


class OpticalFlowBoxTracker:
    """Lightweight display tracker used between expensive DINO refreshes."""

    def __init__(self) -> None:
        self._gray: np.ndarray | None = None
        self._sequence = -1
        self.objects: list[OverlayObject] = []

    def reset(
        self,
        snapshot: FrameSnapshot,
        detections: tuple[TrackedObject, ...],
    ) -> None:
        gray = cv2.cvtColor(snapshot.frame, cv2.COLOR_BGR2GRAY)
        self.objects = [
            OverlayObject(
                id=item.id,
                label=item.label,
                score=item.score,
                bbox=np.array(
                    [item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2],
                    dtype=np.float32,
                ),
            )
            for item in detections
        ]
        for item in self.objects:
            item.points = self._features(gray, item.bbox)
        self._gray = gray
        self._sequence = snapshot.sequence

    def update(self, snapshot: FrameSnapshot) -> None:
        if self._gray is None or snapshot.sequence == self._sequence:
            return
        gray = cv2.cvtColor(snapshot.frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        for item in self.objects:
            if item.points is None or len(item.points) < 2:
                item.points = self._features(self._gray, item.bbox)
            if item.points is None or len(item.points) < 2:
                continue
            moved, status, _ = cv2.calcOpticalFlowPyrLK(
                self._gray,
                gray,
                item.points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    20,
                    0.03,
                ),
            )
            if moved is None or status is None:
                item.points = None
                continue
            valid = status.reshape(-1).astype(bool)
            old_points = item.points.reshape(-1, 2)[valid]
            new_points = moved.reshape(-1, 2)[valid]
            if len(new_points) < 2:
                item.points = None
                continue
            delta = np.median(new_points - old_points, axis=0)
            box_width = max(1.0, float(item.bbox[2] - item.bbox[0]))
            box_height = max(1.0, float(item.bbox[3] - item.bbox[1]))
            if abs(float(delta[0])) > box_width or abs(float(delta[1])) > box_height:
                item.points = None
                continue
            item.bbox[[0, 2]] = np.clip(item.bbox[[0, 2]] + delta[0], 0, width - 1)
            item.bbox[[1, 3]] = np.clip(item.bbox[[1, 3]] + delta[1], 0, height - 1)
            item.points = new_points.reshape(-1, 1, 2).astype(np.float32)
            if len(item.points) < 6:
                refreshed = self._features(gray, item.bbox)
                if refreshed is not None:
                    item.points = refreshed
        self._gray = gray
        self._sequence = snapshot.sequence

    @staticmethod
    def _features(gray: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        height, width = gray.shape
        x1, y1, x2, y2 = (
            max(0, min(width - 1, round(float(value))))
            for value in bbox
        )
        if x2 <= x1 or y2 <= y1:
            return None
        mask = np.zeros_like(gray)
        mask[y1 : y2 + 1, x1 : x2 + 1] = 255
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=30,
            qualityLevel=0.01,
            minDistance=5,
            mask=mask,
            blockSize=5,
        )

    def at_point(self, x: int, y: int) -> OverlayObject | None:
        matches = [
            item
            for item in self.objects
            if item.bbox[0] <= x <= item.bbox[2]
            and item.bbox[1] <= y <= item.bbox[3]
        ]
        if not matches:
            return None
        return min(
            matches,
            key=lambda item: float(
                (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
            ),
        )


@dataclass(frozen=True)
class CommandEvent:
    command: str
    result: AssistiveResult
    snapshot: FrameSnapshot
    audio_total_ms: float
    camera_slot: int = 0


class RealtimeAssistiveRuntime:
    WINDOW_NAME = "Assistive grounding (realtime)"

    def __init__(
        self,
        *,
        camera_index: int,
        camera_indices: tuple[int, ...] | None = None,
        camera_width: int,
        camera_height: int,
        camera_fps: float,
        pipeline: Any,
        processor: AssistiveCommandProcessor,
        wait_for_wake: bool,
        model_loader: Callable[[], None],
        speaker: Callable[[str], None] | None = None,
        disclosure_text: str | None = None,
        post_tts_cooldown_ms: int = 1_200,
        save_frame: Any | None = None,
        target_ui_fps: float = 30.0,
        debug_click_focus: bool = False,
    ) -> None:
        indices = tuple(camera_indices or (camera_index,))
        if not indices:
            indices = (camera_index,)
        self.camera_indices = indices
        self.cameras = [
            ProcessFrameCamera(
                camera_index=index,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
            )
            for index in indices
        ]
        # Backwards-compatible alias used by older callers/tests.
        self.camera = self.cameras[0]
        self.pipeline = pipeline
        self.processor = processor
        self.wait_for_wake = wait_for_wake
        self.model_loader = model_loader
        self.speaker = speaker
        self.disclosure_text = disclosure_text
        self.post_tts_cooldown_ms = post_tts_cooldown_ms
        self.save_frame = save_frame
        self.target_ui_fps = target_ui_fps
        self.debug_click_focus = debug_click_focus
        self.stop_event = threading.Event()
        self.focus_selected = threading.Event()
        self.processor_lock = threading.RLock()
        self.events: queue.Queue[CommandEvent | dict[str, Any]] = queue.Queue()
        self.tracker = OpticalFlowBoxTracker()
        self.status = "STARTING_CAMERA"
        self.selected_object_id: str | None = None
        self._worker: threading.Thread | None = None
        self._latest_result: AssistiveResult | None = None

    def run(self) -> None:
        print(
            json.dumps(
                {"event": "RUNTIME_STARTING", "camera_indices": list(self.camera_indices)},
                ensure_ascii=False,
            ),
            flush=True,
        )
        for camera in self.cameras:
            camera.start()
        for camera in self.cameras:
            camera.latest()
        print(
            json.dumps(
                {"event": "CAMERAS_READY", "camera_fps": [camera.fps for camera in self.cameras]},
                ensure_ascii=False,
            ),
            flush=True,
        )
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)
        self._worker = threading.Thread(
            target=self._command_loop, name="assistive-command", daemon=True
        )
        self._worker.start()
        previous_ui_at = time.monotonic()
        last_save_at = 0.0
        ui_fps = 0.0
        try:
            while not self.stop_event.is_set():
                loop_started = time.monotonic()
                snapshots = [
                    camera.latest(wait_seconds=1.0) for camera in self.cameras
                ]
                self._drain_events()
                # Keep the tracker/display focused on the camera that produced
                # the latest command. Other cameras remain visible for context.
                active_slot = getattr(self, "_active_camera_slot", 0)
                active_slot = min(active_slot, len(snapshots) - 1)
                self.tracker.update(snapshots[active_slot])
                rendered_frames = []
                for slot, snapshot in enumerate(snapshots):
                    view = snapshot.frame.copy()
                    if slot == active_slot:
                        view = self._draw(view, ui_fps, camera_slot=slot)
                    else:
                        cv2.putText(
                            view,
                            f"CAMERA {self.camera_indices[slot]}",
                            (12, 26),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 255),
                            2,
                        )
                    rendered_frames.append(view)
                rendered = (
                    rendered_frames[0]
                    if len(rendered_frames) == 1
                    else np.hstack(rendered_frames)
                )
                if (
                    self.save_frame is not None
                    and loop_started - last_save_at >= 1.0
                ):
                    output_path = Path(self.save_frame)
                    temporary_path = output_path.with_name(
                        f".{output_path.stem}.tmp{output_path.suffix}"
                    )
                    cv2.imwrite(str(temporary_path), rendered)
                    temporary_path.replace(output_path)
                    last_save_at = loop_started
                cv2.imshow(self.WINDOW_NAME, rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                frame_period = 1.0 / max(1.0, self.target_ui_fps)
                remaining = frame_period - (time.monotonic() - loop_started)
                if remaining > 0:
                    time.sleep(remaining)
                now = time.monotonic()
                if now > previous_ui_at:
                    instant = 1.0 / (now - previous_ui_at)
                    ui_fps = instant if ui_fps == 0.0 else ui_fps * 0.9 + instant * 0.1
                previous_ui_at = now
        finally:
            self.stop_event.set()
            self.focus_selected.set()
            for camera in self.cameras:
                camera.stop()
            cv2.destroyWindow(self.WINDOW_NAME)
            if self._worker is not None:
                self._worker.join(timeout=2.0)

    def _command_loop(self) -> None:
        try:
            self.status = "LOADING_GROUNDING_DINO"
            print(json.dumps({"event": "MODEL_LOADING"}), flush=True)
            self.model_loader()
            print(json.dumps({"event": "MODEL_READY"}), flush=True)
            warmup = getattr(self.processor.grounder, "warmup", None)
            if warmup is not None:
                self.status = "WARMING_UP_TEXT_PROMPT_ENCODER"
                print(json.dumps({"event": "GROUNDING_WARMUP"}), flush=True)
                warmup()
            self.status = "GPU_WARMUP"
            warmup = np.zeros((480, 640, 3), dtype=np.uint8)
            self.processor.grounder.detect(warmup, "object")
            print(json.dumps({"event": "GROUNDING_READY"}), flush=True)
            if self.speaker is not None and self.disclosure_text:
                self.status = "AI_VOICE_DISCLOSURE"
                self.speaker(self.disclosure_text)
                if self.post_tts_cooldown_ms:
                    self.status = "TTS_ECHO_COOLDOWN"
                    self.stop_event.wait(self.post_tts_cooldown_ms / 1_000)
            while not self.stop_event.is_set():
                self.status = "LISTENING"
                print(json.dumps({"event": "LISTENING"}), flush=True)
                audio_result = self.pipeline.run_once(wait_for_wake=self.wait_for_wake)
                command = audio_result.transcript
                if not command:
                    self.events.put(audio_result.to_dict())
                    continue
                self.status = "UNDERSTANDING_AND_GROUNDING"
                snapshots = [camera.latest() for camera in self.cameras]
                candidates: list[tuple[int, FrameSnapshot, AssistiveResult]] = []
                for slot, snapshot in enumerate(snapshots):
                    with self.processor_lock:
                        result = self.processor.process(command, snapshot.frame)
                    candidates.append((slot, snapshot, result))

                def result_key(item: tuple[int, FrameSnapshot, AssistiveResult]):
                    result = item[2]
                    grounded = result.state == AssistiveState.GROUNDED
                    score = max((obj.score for obj in result.detections), default=0.0)
                    return (grounded, bool(result.detections), score)

                slot, snapshot, result = max(candidates, key=result_key)
                self._active_camera_slot = slot
                event = CommandEvent(
                    command=command,
                    result=result,
                    snapshot=snapshot,
                    audio_total_ms=audio_result.timings.total_ms,
                    camera_slot=slot,
                )
                self.events.put(event)
                if result.clarification_question and self.speaker is not None:
                    self.status = "SPEAKING_CLARIFICATION"
                    self.speaker(result.clarification_question)
                elif (
                    result.state == AssistiveState.CONVERSATION
                    and result.reply
                    and self.speaker is not None
                ):
                    self.status = "SPEAKING_CONVERSATION"
                    self.speaker(result.reply)
                elif (
                    result.state == AssistiveState.GROUNDED
                    and result.selected_object_id
                    and self.speaker is not None
                ):
                    target = (
                        result.plan.source_object_expression
                        if result.plan is not None
                        and result.plan.source_object_expression
                        else result.plan.target_category
                        if result.plan is not None
                        else "물체"
                    )
                    self.status = "SPEAKING_CONFIRMATION"
                    self.speaker(f"{target}을(를) 찾았습니다.")
                if (
                    self.debug_click_focus
                    and
                    result.state == AssistiveState.CLARIFICATION_REQUIRED
                    and result.detections
                ):
                    self.status = "DEBUG_WAITING_FOR_BB_CLICK"
                    self.focus_selected.clear()
                    while not self.stop_event.is_set():
                        if self.focus_selected.wait(0.1):
                            break
                if result.clarification_question and self.post_tts_cooldown_ms:
                    self.status = "TTS_ECHO_COOLDOWN"
                    self.stop_event.wait(self.post_tts_cooldown_ms / 1_000)
                elif (
                    result.state == AssistiveState.GROUNDED
                    and result.selected_object_id
                    and self.speaker is not None
                    and self.post_tts_cooldown_ms
                ):
                    self.status = "TTS_ECHO_COOLDOWN"
                    self.stop_event.wait(self.post_tts_cooldown_ms / 1_000)
        except Exception as exc:
            self.status = f"ERROR: {type(exc).__name__}"
            self.events.put(
                {
                    "state": "ERROR",
                    "error": type(exc).__name__,
                    "detail": str(exc)[:300],
                }
            )

    def _drain_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                return
            if isinstance(event, dict):
                print(json.dumps(event, ensure_ascii=False, indent=2), flush=True)
                continue
            self._latest_result = event.result
            self.selected_object_id = event.result.selected_object_id
            self.tracker.reset(event.snapshot, event.result.detections)
            output = event.result.to_dict()
            output["utterance"] = event.command
            output["timings_ms"]["audio_to_transcript"] = round(
                event.audio_total_ms, 3
            )
            output["frame_sequence"] = event.snapshot.sequence
            output["camera_slot"] = event.camera_slot
            output["camera_index"] = self.camera_indices[event.camera_slot]
            print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)

    def _on_mouse(self, event, x, y, flags, userdata) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        selected = self.tracker.at_point(x, y)
        if selected is None:
            self.status = "CLICK_INSIDE_A_BB"
            return
        try:
            with self.processor_lock:
                self.processor.set_focus(selected.id)
        except ValueError:
            self.status = "FOCUS_EXPIRED_TRY_AGAIN"
            return
        self.selected_object_id = selected.id
        self.status = f"FOCUS_SELECTED: {selected.id}"
        self.focus_selected.set()
        print(
            json.dumps(
                {
                    "event": "FOCUS_SELECTED",
                    "object_id": selected.id,
                    "label": selected.label,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def _draw(
        self, frame: np.ndarray, ui_fps: float, *, camera_slot: int = 0
    ) -> np.ndarray:
        for item in self.tracker.objects:
            x1, y1, x2, y2 = (round(float(value)) for value in item.bbox)
            selected = item.id == self.selected_object_id
            color = (0, 255, 0) if selected else (0, 180, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if selected else 2)
            cv2.putText(
                frame,
                f"{item.id} {item.label} {item.score:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        cv2.putText(
            frame,
            f"CAM {self.camera_indices[camera_slot]} "
            f"{self.cameras[camera_slot].fps:4.1f} FPS  "
            f"UI {ui_fps:4.1f} FPS",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            self.status,
            (12, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        return frame
