from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    color: np.ndarray
    depth: np.ndarray | None
    timestamp_ms: int
    camera_info: dict[str, Any]


class RealSenseSource:
    """Direct pyrealsense2 RGB-D source; do not run realsense2_camera concurrently."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2가 없습니다. RealSense SDK Python 바인딩을 설치하거나 "
                "--camera opencv/--video를 사용하세요."
            ) from exc
        self.rs = rs
        self.pipeline = None
        profile = None
        failures: list[str] = []
        for color_size, depth_size, candidate_fps in _profile_candidates(width, height, fps):
            pipeline = rs.pipeline()
            config = rs.config()
            label = "SDK default"
            if color_size is None or depth_size is None:
                config.enable_stream(rs.stream.color)
                config.enable_stream(rs.stream.depth)
            else:
                cw, ch = color_size
                dw, dh = depth_size
                label = f"color={cw}x{ch}, depth={dw}x{dh}, fps={candidate_fps}"
                config.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, candidate_fps)
                config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, candidate_fps)
            try:
                profile = pipeline.start(config)
                self.pipeline = pipeline
                self.profile_label = label
                break
            except RuntimeError as exc:
                failures.append(f"{label}: {exc}")
        if profile is None or self.pipeline is None:
            detail = "; ".join(failures)
            raise RuntimeError(
                "RealSense RGB-D 스트림을 시작할 수 없습니다. 다른 realsense2_camera/"
                "Python 프로세스가 카메라를 점유하는지, USB 3 포트인지, udev 권한이 "
                f"설정됐는지 확인하세요. 시도 결과: {detail}")
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    def read(self) -> CameraFrame | None:
        # A newly started D435I can briefly produce an incomplete frameset.
        # Treat that as camera warm-up, not end-of-stream.
        for _ in range(5):
            try:
                frames = self.align.process(
                    self.pipeline.wait_for_frames(timeout_ms=3000))
            except RuntimeError:
                continue
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            return CameraFrame(
                color=np.asanyarray(color_frame.get_data()),
                depth=np.asanyarray(depth_frame.get_data()),
                timestamp_ms=round(color_frame.get_timestamp()),
                camera_info={
                    "width": intr.width, "height": intr.height,
                    "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
                    "coeffs": list(intr.coeffs), "depth_scale": self.depth_scale,
                },
            )
        return None

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()


def _profile_candidates(width: int, height: int, fps: int):
    """Common cross-device profiles, ending with librealsense defaults."""
    requested = ((width, height), (width, height), fps)
    values = [
        requested,
        ((1280, 720), (848, 480), 30),
        ((848, 480), (848, 480), 30),
        ((640, 480), (640, 480), 30),
        ((640, 480), (640, 480), 15),
        (None, None, 0),
    ]
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return tuple(unique)


class OpenCVSource:
    """Development fallback for a video file or ordinary camera."""

    def __init__(self, source: str | int = 0) -> None:
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"영상 입력을 열 수 없습니다: {source}")

    def read(self) -> CameraFrame | None:
        import time
        ok, color = self.cap.read()
        if not ok:
            return None
        return CameraFrame(color, None, round(time.time() * 1000), {})

    def close(self) -> None:
        self.cap.release()


class SyntheticSource:
    """No-hardware UI source used for text/offline interface development."""

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height

    def read(self) -> CameraFrame:
        import cv2
        import time
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[:] = (12, 10, 5)
        for radius, alpha in ((270, .12), (190, .18), (110, .28)):
            layer = image.copy()
            cv2.circle(layer, (self.width // 2, self.height // 2), radius,
                       (210, 130, 25), 2)
            image = cv2.addWeighted(layer, alpha, image, 1 - alpha, 0)
        cv2.putText(image, "JARVIS OPTICS / OFFLINE TEST FEED",
                    (self.width // 2 - 260, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (210, 220, 95), 1, cv2.LINE_AA)
        cv2.putText(image, "Connect RealSense for live RGB-D perception",
                    (self.width // 2 - 235, self.height // 2 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (115, 130, 85), 1, cv2.LINE_AA)
        time.sleep(1 / 20)
        return CameraFrame(image, None, round(time.time() * 1000), {})

    def close(self) -> None:
        pass
