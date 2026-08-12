from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BufferedFrame:
    timestamp_ms: int
    jpeg: bytes
    quality: float = 1.0
    sharpness: float = 1.0
    depth_valid_ratio: float | None = None
    foreground_ratio: float | None = None


class RecentFrameBuffer:
    """Memory-bounded JPEG history used for temporal questions, not MP4 replay."""

    def __init__(self, seconds: float = 8.0, sample_fps: float = 4.0,
                 jpeg_quality: int = 82) -> None:
        self.seconds = seconds
        self.sample_period_ms = round(1000 / sample_fps)
        self.jpeg_quality = jpeg_quality
        self.frames: deque[BufferedFrame] = deque()
        self._last_ms = -10**15

    def add(self, frame: np.ndarray, timestamp_ms: int | None = None,
            depth: np.ndarray | None = None, depth_scale: float | None = None) -> None:
        now = timestamp_ms if timestamp_ms is not None else round(time.time() * 1000)
        if now - self._last_ms < self.sample_period_ms:
            return
        ok, encoded = cv2.imencode(".jpg", frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        sharpness, valid_ratio, foreground_ratio, quality = _frame_quality(
            frame, depth, depth_scale)
        self.frames.append(BufferedFrame(
            now, encoded.tobytes(), quality, sharpness, valid_ratio, foreground_ratio))
        self._last_ms = now
        cutoff = now - round(self.seconds * 1000)
        while self.frames and self.frames[0].timestamp_ms < cutoff:
            self.frames.popleft()

    def representative(self, count: int = 5) -> list[BufferedFrame]:
        values = list(self.frames)
        if len(values) <= count:
            return values
        # Preserve temporal coverage, but select the clearest/non-occluded item
        # inside each time bucket instead of blindly sampling blurred frames.
        boundaries = np.linspace(0, len(values), count + 1).round().astype(int)
        selected = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            bucket = values[start:max(start + 1, stop)]
            selected.append(max(bucket, key=lambda item: item.quality))
        return sorted(selected, key=lambda item: item.timestamp_ms)


def _frame_quality(frame: np.ndarray, depth: np.ndarray | None,
                   depth_scale: float | None) -> tuple[float, float | None, float | None, float]:
    """Cheap RGB-D quality gate; it does not claim semantic hand detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = min(1.0, variance / 180.0)
    if depth is None or depth.size == 0:
        return sharpness, None, None, sharpness
    valid = depth > 0
    valid_ratio = float(valid.mean())
    foreground_ratio = 0.0
    if valid.any():
        values = depth[valid].astype(np.float32)
        # The nearest decile is a conservative proxy for a large hand/tool in
        # front of the work area. It is only a quality penalty, never evidence.
        near_threshold = float(np.percentile(values, 10))
        foreground_ratio = float(((depth > 0) & (depth <= near_threshold)).mean())
    depth_score = min(1.0, valid_ratio / 0.65)
    occlusion_score = max(0.0, 1.0 - max(0.0, foreground_ratio - 0.18) * 3.0)
    quality = 0.55 * sharpness + 0.30 * depth_score + 0.15 * occlusion_score
    return sharpness, valid_ratio, foreground_ratio, float(max(0.0, min(1.0, quality)))
