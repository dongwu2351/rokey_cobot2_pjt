from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .assistive_models import BoundingBox, TrackedObject


@dataclass(frozen=True)
class VisualMatch:
    canonical_name: str
    similarity: float
    sample_path: str | None


class VisualMemory:
    """Small persistent RGB appearance memory for webcam proof-of-concept.

    This is intentionally an appearance hint, not a robot pose store. Current
    BBoxes are always obtained from the current frame before a robot action.
    """

    def __init__(self, path: str | Path, *, sample_dir: str | Path | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_dir = Path(sample_dir or self.path.with_name("visual_samples"))
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS visual_samples (
                sample_id TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL,
                descriptor_json TEXT NOT NULL,
                sample_path TEXT,
                created_at_ms INTEGER NOT NULL,
                last_used_ms INTEGER NOT NULL
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def remember(self, canonical_name: str, image: Any, bbox: BoundingBox) -> str:
        crop = crop_image(image, bbox)
        descriptor = appearance_descriptor(crop)
        sample_id = uuid.uuid4().hex
        sample_path = self.sample_dir / f"{sample_id}.jpg"
        crop.save(sample_path, quality=88)
        now = round(time.time() * 1_000)
        self._connection.execute(
            "INSERT INTO visual_samples VALUES (?, ?, ?, ?, ?, ?)",
            (sample_id, canonical_name, json.dumps(descriptor), str(sample_path), now, now),
        )
        self._connection.commit()
        return sample_id

    def best_match(
        self,
        canonical_name: str,
        image: Any,
        objects: tuple[TrackedObject, ...] | list[TrackedObject],
        *,
        minimum_similarity: float = 0.82,
    ) -> tuple[TrackedObject, VisualMatch] | None:
        rows = self._connection.execute(
            "SELECT canonical_name, descriptor_json, sample_path FROM visual_samples "
            "WHERE canonical_name = ? ORDER BY last_used_ms DESC",
            (canonical_name,),
        ).fetchall()
        if not rows:
            return None
        best: tuple[TrackedObject, float, str | None] | None = None
        for obj in objects:
            score = max(
                cosine_similarity(appearance_descriptor(crop_image(image, obj.bbox)), json.loads(row[1]))
                for row in rows
            )
            if best is None or score > best[1]:
                best = (obj, score, rows[0][2])
        if best is None or best[1] < minimum_similarity:
            return None
        now = round(time.time() * 1_000)
        self._connection.execute(
            "UPDATE visual_samples SET last_used_ms = ? WHERE canonical_name = ?",
            (now, canonical_name),
        )
        self._connection.commit()
        return best[0], VisualMatch(canonical_name, best[1], best[2])


def crop_image(image: Any, bbox: BoundingBox) -> Image.Image:
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be an HxWx3 array")
        # OpenCV frames are BGR; convert to RGB before storing the sample.
        pil = Image.fromarray(image[:, :, ::-1].copy()).convert("RGB")
    else:
        raise TypeError("image must be a PIL image or NumPy array")
    width, height = pil.size
    left = max(0, min(width - 1, int(bbox.x1)))
    top = max(0, min(height - 1, int(bbox.y1)))
    right = max(left + 1, min(width, int(bbox.x2)))
    bottom = max(top + 1, min(height, int(bbox.y2)))
    return pil.crop((left, top, right, bottom))


def appearance_descriptor(image: Image.Image) -> list[float]:
    """Compact, dependency-light appearance descriptor for the first prototype."""
    thumbnail = image.resize((24, 24), Image.Resampling.BILINEAR)
    values = np.asarray(thumbnail, dtype=np.float32).reshape(-1) / 255.0
    values -= float(values.mean())
    norm = float(np.linalg.norm(values))
    if norm == 0:
        return [0.0] * values.size
    return (values / norm).round(6).tolist()


def cosine_similarity(first: list[float], second: list[float]) -> float:
    a, b = np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)
    if a.shape != b.shape or not a.size:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0
