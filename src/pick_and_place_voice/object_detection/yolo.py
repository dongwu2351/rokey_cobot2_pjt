########## YoloModel ##########
import os
import json
import time
from collections import Counter
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO
import numpy as np


PACKAGE_NAME = "pick_and_place_voice"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

# Detector trained on this cell (2026-08-10): four classes, because the bench
# holds two screwdrivers that must be told apart and no drill or pliers at all.
# Retrained on 1016 four-camera captures with 439 images human-verified; on a
# human-checked validation set it scores mAP50 0.859 against the shipped
# model's 0.652, and finds the blue wrench the old one never saw (0.394->0.774).
#
# The previous model and its class map are still here: set YOLO_MODEL and
# YOLO_CLASSES to roll back without editing code, e.g.
#   YOLO_MODEL=yolov8n_tools_0122.pt YOLO_CLASSES=class_name_tool.json ros2 run ...
YOLO_MODEL_FILENAME = os.environ.get("YOLO_MODEL", "tools_v5_4class.pt")
YOLO_CLASS_NAME_JSON = os.environ.get("YOLO_CLASSES", "class_name_tool_v5.json")

def _resolve_resource(filename, fallback_relative_paths=()):
    """Resolve an installed resource, then source-workspace fallbacks."""
    candidates = [
        Path(PACKAGE_PATH) / "resource" / filename,
        Path(__file__).resolve().parents[1] / "resource" / filename,
    ]
    workspace = Path(__file__).resolve().parents[2]
    candidates.extend(workspace / relative for relative in fallback_relative_paths)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    checked = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{filename} not found. Checked:\n  - {checked}")


YOLO_MODEL_PATH = _resolve_resource(
    YOLO_MODEL_FILENAME,
    ("pick_and_place_text/resource/yolov8n_tools_0122.pt",),
)
YOLO_JSON_PATH = _resolve_resource(
    YOLO_CLASS_NAME_JSON,
    ("pick_and_place_text/resource/class_name_tool.json",),
)


class YoloModel:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)
        with open(YOLO_JSON_PATH, "r", encoding="utf-8") as file:
            class_dict = json.load(file)
            self.reversed_class_dict = {v: int(k) for k, v in class_dict.items()}
        self.class_names = tuple(self.reversed_class_dict)

    def track_frame(self, frame, confidence_threshold=0.5):
        """Detections carrying a persistent id per object.

        Plain detection answers "where are the hammers", which is not the same
        question as "where is the hammer I was following". With two of them in
        view, or a frame where the one being followed scores lower, picking the
        best-scoring box every frame silently swaps target. ByteTrack keeps an
        id attached to each object across frames, and it also bridges the gap
        when a box drops out for a frame or two, so the id survives a blink.

        `persist=True` is what carries tracker state between calls - without it
        every call starts a new tracker and ids restart from 1 each frame.
        Detections with no id (a track too young to be confirmed) come back
        with id None and the caller falls back to matching by name.
        """
        if frame is None:
            return []

        result = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml", verbose=False
        )[0]
        ids = result.boxes.id
        ids = [None] * len(result.boxes) if ids is None else [
            int(v) for v in ids.tolist()
        ]
        detections = []
        for box, score, label, track_id in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
            ids,
        ):
            if score < confidence_threshold:
                continue
            label_id = int(label)
            detections.append(
                {
                    "box": [float(value) for value in box],
                    "score": float(score),
                    "label": label_id,
                    "name": result.names[label_id],
                    "id": track_id,
                }
            )
        return detections

    def predict_frame(self, frame, confidence_threshold=0.5):
        """Return drawable detections for one BGR frame."""
        if frame is None:
            return []

        result = self.model(frame, verbose=False)[0]
        detections = []
        for box, score, label in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            if score < confidence_threshold:
                continue
            label_id = int(label)
            detections.append(
                {
                    "box": [float(value) for value in box],
                    "score": float(score),
                    "label": label_id,
                    "name": result.names[label_id],
                }
            )
        return detections

    def get_frames(self, img_node, duration=1.0):
        """get frames while target_time"""
        end_time = time.time() + duration
        frames = {}

        while time.time() < end_time:
            rclpy.spin_once(img_node)
            frame = img_node.get_color_frame()
            stamp = img_node.get_color_frame_stamp()
            if frame is not None:
                frames[stamp] = frame
            time.sleep(0.01)

        if not frames:
            print("No frames captured in %.2f seconds", duration)

        print("%d frames captured", len(frames))
        return list(frames.values())

    def get_best_detection(self, img_node, target):
        rclpy.spin_once(img_node)
        frames = self.get_frames(img_node)
        if not frames:  # Check if frames are empty
            return None, None

        results = self.model(frames, verbose=False)
        print("classes: ")
        print(results[0].names)
        detections = self._aggregate_detections(results)
        label_id = self.reversed_class_dict.get(target)
        if label_id is None:
            print(f"Unknown target: {target}")
            return None, None
        print("label_id: ", label_id)
        print("detections: ", detections)

        matches = [d for d in detections if d["label"] == label_id]
        if not matches:
            print("No matches found for the target label.")
            return None, None
        best_det = max(matches, key=lambda x: x["score"])
        return best_det["box"], best_det["score"]

    def _aggregate_detections(self, results, confidence_threshold=0.5, iou_threshold=0.5):
        """
        Fuse raw detection boxes across frames using IoU-based grouping
        and majority voting for robust final detections.
        """
        raw = []
        for res in results:
            for box, score, label in zip(
                res.boxes.xyxy.tolist(),
                res.boxes.conf.tolist(),
                res.boxes.cls.tolist(),
            ):
                if score >= confidence_threshold:
                    raw.append({"box": box, "score": score, "label": int(label)})

        final = []
        used = [False] * len(raw)

        for i, det in enumerate(raw):
            if used[i]:
                continue
            group = [det]
            used[i] = True
            for j, other in enumerate(raw):
                if not used[j] and other["label"] == det["label"]:
                    if self._iou(det["box"], other["box"]) >= iou_threshold:
                        group.append(other)
                        used[j] = True

            boxes = np.array([g["box"] for g in group])
            scores = np.array([g["score"] for g in group])
            labels = [g["label"] for g in group]

            final.append(
                {
                    "box": boxes.mean(axis=0).tolist(),
                    "score": float(scores.mean()),
                    "label": Counter(labels).most_common(1)[0][0],
                }
            )

        return final

    def _iou(self, box1, box2):
        """
        Compute Intersection over Union (IoU) between two boxes [x1, y1, x2, y2].
        """
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
