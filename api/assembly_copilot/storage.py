from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2


class SessionStore:
    def __init__(self, root: str | Path, session_id: str, metadata: dict[str, Any]) -> None:
        self.root = Path(root).resolve() / session_id
        self.keyframes = self.root / "keyframes"
        self.video_dir = self.root / "video"
        self.observations = self.root / "observations"
        for path in (self.keyframes, self.video_dir, self.observations):
            path.mkdir(parents=True, exist_ok=True)
        (self.root / "session.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self.events_path = self.root / "events.jsonl"
        self.writer = None

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        import time
        row = {"timestamp_ms": round(time.time() * 1000), "type": kind, **payload}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def save_keyframe(self, frame, name: str) -> str:
        path = self.keyframes / f"{name}.jpg"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"이미지 저장 실패: {path}")
        self.event("KEYFRAME", {"path": str(path)})
        return str(path)

    def start_video(self, frame, fps: float = 30.0) -> str:
        if self.writer is not None:
            return str(self.video_path)
        h, w = frame.shape[:2]
        self.video_path = self.video_dir / "color.mp4"
        self.writer = cv2.VideoWriter(str(self.video_path),
                                      cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not self.writer.isOpened():
            self.writer = None
            raise RuntimeError(f"영상 저장 실패: {self.video_path}")
        self.event("VIDEO_START", {"path": str(self.video_path)})
        return str(self.video_path)

    def write_video(self, frame) -> None:
        if self.writer is not None:
            self.writer.write(frame)

    def save_observation(self, data: dict[str, Any], name: str) -> str:
        path = self.observations / f"{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.event("VIDEO_STOP", {"path": str(self.video_path)})
            self.writer = None
