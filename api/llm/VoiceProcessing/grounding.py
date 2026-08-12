from __future__ import annotations

import itertools
import inspect
import time
from typing import Any, Protocol, Sequence

from .assistive_models import BoundingBox, Detection, TrackedObject


class OpenVocabularyGrounder(Protocol):
    def detect(self, image: Any, query: str) -> Sequence[Detection]: ...


class YOLOEBackend:
    """YOLOE-26 segmentation backend with dynamic text prompts."""

    def __init__(
        self,
        *,
        model_path: str = "yoloe-26s-seg.pt",
        confidence: float = 0.15,
        image_size: int = 640,
        device: str | None = None,
    ) -> None:
        if not 0 <= confidence <= 1 or image_size < 32:
            raise ValueError("invalid YOLOE configuration")
        self.model_path = model_path
        self.confidence = confidence
        self.image_size = image_size
        self.requested_device = device
        self._model = None
        self._classes: tuple[str, ...] = ()

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise RuntimeError(
                "YOLOE requires ultralytics; install VoiceProcessing requirements"
            ) from exc
        self._model = YOLOE(self.model_path)

    @staticmethod
    def _prompt_classes(query: str) -> tuple[str, ...]:
        classes = tuple(
            dict.fromkeys(
                part.strip().strip(".,")
                for part in query.replace("\n", ".").split(".")
                if part.strip().strip(".,")
            )
        )
        if not classes:
            raise ValueError("YOLOE query must not be empty")
        return classes[:20]

    def detect(self, image: Any, query: str) -> tuple[Detection, ...]:
        self.load()
        classes = self._prompt_classes(query)
        if classes != self._classes:
            self._model.set_classes(list(classes))
            self._classes = classes
        kwargs = {
            "source": image,
            "conf": self.confidence,
            "imgsz": self.image_size,
            "verbose": False,
        }
        if self.requested_device:
            kwargs["device"] = self.requested_device
        result = self._model.predict(**kwargs)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return ()
        names = result.names or dict(enumerate(classes))
        boxes = result.boxes.xyxy.detach().cpu().tolist()
        scores = result.boxes.conf.detach().cpu().tolist()
        labels = result.boxes.cls.detach().cpu().tolist()
        masks = result.masks.data.detach().cpu() if result.masks is not None else None
        height, width = image.shape[:2]
        detections: list[Detection] = []
        for index, (box, score, class_index) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = box
            attributes: dict[str, Any] = {"segmentation": masks is not None}
            if masks is not None:
                attributes["mask_area_ratio"] = float(masks[index].float().mean())
            detections.append(
                Detection(
                    label=str(names.get(int(class_index), classes[int(class_index)])),
                    score=float(score),
                    bbox=BoundingBox(
                        x1=max(0.0, min(float(width), x1)),
                        y1=max(0.0, min(float(height), y1)),
                        x2=max(0.0, min(float(width), x2)),
                        y2=max(0.0, min(float(height), y2)),
                    ),
                    attributes=attributes,
                )
            )
        return tuple(detections)

    def warmup(self) -> None:
        """Load the text encoder before the microphone starts listening."""
        self.load()
        if not self._classes:
            self._model.set_classes(["object"])
            self._classes = ("object",)


class FallbackGrounder:
    """Use a fast primary detector and call the precise fallback only on miss."""

    def __init__(
        self,
        primary: OpenVocabularyGrounder,
        fallback: OpenVocabularyGrounder,
        *,
        minimum_primary_score: float = 0.30,
    ):
        if not 0 <= minimum_primary_score <= 1:
            raise ValueError("minimum_primary_score must be between zero and one")
        self.primary = primary
        self.fallback = fallback
        self.minimum_primary_score = minimum_primary_score

    def load(self) -> None:
        loader = getattr(self.primary, "load", None)
        if loader is not None:
            loader()

    def warmup(self) -> None:
        loader = getattr(self.primary, "warmup", None)
        if loader is not None:
            loader()

    def detect(self, image: Any, query: str) -> Sequence[Detection]:
        detections = tuple(self.primary.detect(image, query))
        if detections and max(item.score for item in detections) >= self.minimum_primary_score:
            return detections
        return tuple(self.fallback.detect(image, query))


class GroundingDINOBackend:
    """Lazy Hugging Face Grounding DINO adapter; model imports are optional."""

    def __init__(
        self,
        *,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        max_box_area_ratio: float = 0.9,
        device: str | None = None,
    ) -> None:
        if (
            not 0 <= box_threshold <= 1
            or not 0 <= text_threshold <= 1
            or not 0 < max_box_area_ratio <= 1
        ):
            raise ValueError("grounding thresholds must be between zero and one")
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_box_area_ratio = max_box_area_ratio
        self.requested_device = device
        self._processor = None
        self._model = None
        self._device = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Grounding DINO requires the packages in requirements-vision.txt"
            ) from exc
        device = self.requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(device)
        self._model.eval()
        self._device = device

    def detect(self, image: Any, query: str) -> tuple[Detection, ...]:
        prompt = query.strip()
        if not prompt:
            raise ValueError("grounding query must not be empty")
        if not prompt.endswith("."):
            prompt += "."
        self.load()
        import torch
        from PIL import Image

        if not isinstance(image, Image.Image):
            try:
                import cv2
                import numpy as np
            except ImportError as exc:
                raise RuntimeError("NumPy/OpenCV image conversion is unavailable") from exc
            if not isinstance(image, np.ndarray):
                raise TypeError("image must be a PIL image or NumPy array")
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(
            self._device
        )
        with torch.inference_mode():
            outputs = self._model(**inputs)
        post_process = self._processor.post_process_grounded_object_detection
        # Transformers 5 renamed ``box_threshold`` to ``threshold``.  Select
        # the supported keyword so installations on either side of that API
        # change remain usable.
        threshold_name = (
            "box_threshold"
            if "box_threshold" in inspect.signature(post_process).parameters
            else "threshold"
        )
        result = post_process(
            outputs,
            inputs.input_ids,
            **{
                threshold_name: self.box_threshold,
                "text_threshold": self.text_threshold,
                "target_sizes": [(image.height, image.width)],
            },
        )[0]
        # Avoid touching the deprecated ``labels`` key when ``text_labels``
        # is available; Transformers emits a warning even for a fallback get.
        labels = (
            result["text_labels"]
            if "text_labels" in result
            else result.get("labels", [])
        )
        detections: list[Detection] = []
        image_area = float(image.width * image.height)
        for box, score, label in zip(result["boxes"], result["scores"], labels):
            normalized_label = str(label).strip()
            box_area = max(0.0, float(box[2] - box[0])) * max(
                0.0, float(box[3] - box[1])
            )
            if (
                not normalized_label
                or float(box[2]) <= float(box[0])
                or float(box[3]) <= float(box[1])
                or box_area / image_area > self.max_box_area_ratio
            ):
                continue
            detections.append(
                Detection(
                    label=normalized_label,
                    score=float(score),
                    bbox=BoundingBox(
                        x1=max(0.0, float(box[0])),
                        y1=max(0.0, float(box[1])),
                        x2=min(float(image.width), float(box[2])),
                        y2=min(float(image.height), float(box[3])),
                    ),
                )
            )
        deduplicated: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.score, reverse=True):
            if any(_iou(detection.bbox, kept.bbox) >= 0.7 for kept in deduplicated):
                continue
            deduplicated.append(detection)
        return tuple(_suppress_multi_object_containers(deduplicated))


def _iou(first: BoundingBox, second: BoundingBox) -> float:
    x1, y1 = max(first.x1, second.x1), max(first.y1, second.y1)
    x2, y2 = min(first.x2, second.x2), min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    first_area = (first.x2 - first.x1) * (first.y2 - first.y1)
    second_area = (second.x2 - second.x1) * (second.y2 - second.y1)
    return intersection / (first_area + second_area - intersection)


def _suppress_multi_object_containers(
    detections: Sequence[Detection],
) -> list[Detection]:
    """Drop a weak box that merely wraps two stronger individual objects."""
    kept: list[Detection] = []
    for candidate in detections:
        contained_stronger = 0
        for other in detections:
            if other is candidate or other.score < candidate.score:
                continue
            center_x, center_y = other.bbox.center
            if (
                candidate.bbox.x1 <= center_x <= candidate.bbox.x2
                and candidate.bbox.y1 <= center_y <= candidate.bbox.y2
            ):
                contained_stronger += 1
        if contained_stronger < 2:
            kept.append(candidate)
    return kept


class SceneTracker:
    """Small IoU tracker that preserves IDs between nearby webcam frames."""

    def __init__(self, *, minimum_iou: float = 0.3, retention_ms: int = 2_000) -> None:
        if not 0 < minimum_iou <= 1 or retention_ms <= 0:
            raise ValueError("invalid tracker configuration")
        self.minimum_iou = minimum_iou
        self.retention_ms = retention_ms
        self._counter = itertools.count(1)
        self._objects: dict[str, TrackedObject] = {}

    def update(
        self,
        detections: Sequence[Detection],
        *,
        now_ms: int | None = None,
    ) -> tuple[TrackedObject, ...]:
        timestamp = now_ms if now_ms is not None else round(time.time() * 1_000)
        unmatched_ids = set(self._objects)
        updated: list[TrackedObject] = []
        for detection in sorted(detections, key=lambda item: item.score, reverse=True):
            candidates = [
                (object_id, _iou(detection.bbox, self._objects[object_id].bbox))
                for object_id in unmatched_ids
                if self._objects[object_id].label == detection.label
            ]
            object_id = None
            if candidates:
                best_id, best_iou = max(candidates, key=lambda item: item[1])
                if best_iou >= self.minimum_iou:
                    object_id = best_id
                    unmatched_ids.remove(best_id)
            if object_id is None:
                object_id = f"object_{next(self._counter):04d}"
            tracked = TrackedObject(
                id=object_id,
                label=detection.label,
                score=detection.score,
                bbox=detection.bbox,
                attributes=detection.attributes,
                last_seen_ms=timestamp,
            )
            self._objects[object_id] = tracked
            updated.append(tracked)
        self._objects = {
            object_id: item
            for object_id, item in self._objects.items()
            if timestamp - item.last_seen_ms <= self.retention_ms
        }
        return tuple(updated)

    def get(self, object_id: str | None) -> TrackedObject | None:
        return self._objects.get(object_id) if object_id else None

    def visible(self) -> tuple[TrackedObject, ...]:
        return tuple(self._objects.values())
