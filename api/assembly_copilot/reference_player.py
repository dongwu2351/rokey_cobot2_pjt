from __future__ import annotations

import time
from pathlib import Path

import cv2


class ReferenceVideoPlayer:
    """Non-blocking manual clip reader for the AR reference panel."""

    def __init__(self) -> None:
        self.cap = None
        self.path = None
        self.start_s = self.end_s = 0.0
        self.started = 0.0

    def play(self, item: dict) -> None:
        self.close()
        path = str(item["file"])
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"참고 영상을 열 수 없습니다: {path}")
        self.path = path
        self.start_s = float(item.get("start_s", 0.0))
        self.end_s = float(item.get("end_s", 0.0))
        cap.set(cv2.CAP_PROP_POS_MSEC, self.start_s * 1000)
        self.cap = cap
        self.started = time.monotonic()

    def frame(self):
        if self.cap is None:
            return None
        elapsed_s = time.monotonic() - self.started
        target_s = self.start_s + elapsed_s
        if self.end_s > self.start_s and target_s >= self.end_s:
            self.close(); return None
        self.cap.set(cv2.CAP_PROP_POS_MSEC, target_s * 1000)
        ok, frame = self.cap.read()
        if not ok:
            self.close(); return None
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None
