# YoloModel
import json
import time
from collections import Counter
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO
import numpy as np


PACKAGE_NAME = "object_detection"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)
RESOURCE_CONFIGS = {
    "tools": {
        "model": "yolov8n_tools_0122.pt",
        "classes": "class_name_tool.json",
        "model_fallbacks": (
            "pick_and_place_text/resource/yolov8n_tools_0122.pt",
        ),
        "class_fallbacks": (
            "pick_and_place_text/resource/class_name_tool.json",
        ),
    },
    "fruits": {
        "model": "fruits_best.pt",
        "classes": "class_name_fruit.json",
        "model_fallbacks": ("fruits/best.pt",),
        "class_fallbacks": (),
    },
}


def _resolve_resource(filename, fallback_relative_paths=()):
    """Resolve a resource from install-space or the source workspace."""
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


class YoloModel:
    def __init__(self, target_set="tools"):
        if target_set not in RESOURCE_CONFIGS:
            choices = ", ".join(RESOURCE_CONFIGS)
            raise ValueError(
                f"Unknown target_set '{target_set}'. Choose one of: {choices}"
            )
        config = RESOURCE_CONFIGS[target_set]
        self.target_set = target_set
        self.model_path = _resolve_resource(
            config["model"], config["model_fallbacks"]
        )
        self.class_path = _resolve_resource(
            config["classes"], config["class_fallbacks"]
        )
        self.model = YOLO(self.model_path)
        with open(self.class_path, "r", encoding="utf-8") as file:
            class_dict = json.load(file)
            self.reversed_class_dict = {
                value.lower(): int(key)
                for key, value in class_dict.items()
            }
        self.class_names = tuple(self.reversed_class_dict)
        model_names = tuple(
            str(self.model.names[index]).lower()
            for index in sorted(self.model.names)
        )
        if model_names != self.class_names:
            raise ValueError(
                "Model/class JSON mismatch: "
                f"model={model_names}, json={self.class_names}"
            )

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
                    "name": str(result.names[label_id]).lower(),
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
