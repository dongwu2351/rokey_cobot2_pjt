#!/usr/bin/env python3
"""Point at a tool and watch its box light up.

The rig triangulates two points on your index finger into a 3D ray, and the
detector triangulates each tool into a 3D position. The tool whose position
sits at the smallest ANGLE off that ray is the one you mean - the same
quantity your eye uses, which is why it feels right rather than fiddly.

This is the selection half of "이거 잡아봐": once a tool is selected, the
robot already knows how to fetch it by class name. Nothing moves here.

    python3 tools/point_to_select.py
    python3 tools/point_to_select.py --no-gesture   # any hand pose points

Keys: SPACE freeze | S save | Q quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "pick_and_place_voice"))

from object_detection.webcam_rig import WebcamRig, HandIntruderDetector  # noqa: E402
from object_detection import pointing                                    # noqa: E402

CONF = 0.30
IMGSZ = 960
CROP_MARGIN_PX = 20
CROP_MAX_SCALE = 3.0
MAX_RAY_GAP_M = 0.040
TARGET_Z_MM = (-40.0, 160.0)
WORKSPACE_CORNERS_M = [np.array([x, y, z])
                       for x in (0.12, 0.78) for y in (-0.45, 0.45)
                       for z in (0.0, 0.25)]
QUAD = (640, 360)
COLOR_IDLE = (150, 150, 150)
COLOR_PICK = (80, 255, 120)
COLOR_RAY = (0, 220, 255)
COLOR_AMBIG = (0, 200, 255)


def workspace_crop(camera, shape):
    height, width = shape[:2]
    pixels = [p for p in camera.project(WORKSPACE_CORNERS_M) if p is not None]
    if len(pixels) < 4:
        return 0, 0, width, height, 1.0
    xs, ys = [p[0] for p in pixels], [p[1] for p in pixels]
    x1 = int(max(0, min(xs) - CROP_MARGIN_PX))
    y1 = int(max(0, min(ys) - CROP_MARGIN_PX))
    x2 = int(min(width, max(xs) + CROP_MARGIN_PX))
    y2 = int(min(height, max(ys) + CROP_MARGIN_PX))
    if x2 - x1 < 40 or y2 - y1 < 40:
        return 0, 0, width, height, 1.0
    return x1, y1, x2, y2, min(CROP_MAX_SCALE, 1280.0 / max(1, x2 - x1))


def detect(model, frames, crops):
    """{cam: {class: (box_full_frame, conf)}} - best box per class per view."""
    cams = list(frames)
    images = []
    for cam in cams:
        x1, y1, x2, y2, scale = crops[cam]
        crop = frames[cam][y1:y2, x1:x2]
        if scale > 1.01:
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        images.append(crop)
    out = {}
    for cam, result in zip(cams, model(images, verbose=False, conf=CONF,
                                       imgsz=IMGSZ)):
        x1, y1, _, _, scale = crops[cam]
        best = {}
        for box, conf, cls in zip(result.boxes.xyxy.tolist(),
                                  result.boxes.conf.tolist(),
                                  result.boxes.cls.tolist()):
            cls = int(cls)
            full = [box[0] / scale + x1, box[1] / scale + y1,
                    box[2] / scale + x1, box[3] / scale + y1]
            if cls not in best or conf > best[cls][1]:
                best[cls] = (full, float(conf))
        out[cam] = best
    return out


def locate_objects(rig, per_cam):
    """{class: (position_mm, cams)} for classes two cameras agree on."""
    classes = set()
    for dets in per_cam.values():
        classes |= set(dets)
    located = {}
    for cls in classes:
        picks = {cam: ((d[cls][0][0] + d[cls][0][2]) / 2.0,
                       (d[cls][0][1] + d[cls][0][3]) / 2.0)
                 for cam, d in per_cam.items() if cls in d}
        if len(picks) < 2:
            continue
        point, gap = rig.triangulate(picks)
        if point is None or gap is None or gap > MAX_RAY_GAP_M:
            continue
        position = np.asarray(point) * 1000.0
        if not TARGET_Z_MM[0] <= position[2] <= TARGET_Z_MM[1]:
            continue                      # not a tool lying on the table
        located[cls] = (position, sorted(picks))
    return located


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--no-gesture", action="store_true",
                        help="do not require the index to be extended")
    parser.add_argument("--max-angle", type=float,
                        default=pointing.MAX_POINT_ANGLE_DEG)
    args = parser.parse_args()

    from ultralytics import YOLO
    weights = args.weights or sorted(
        (Path.home() / "tool_dataset" / "runs").glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime)[-1]
    model = YOLO(str(weights))
    model.to("cuda")
    names = [line for line in (Path.home() / "tool_dataset" / "labeled"
                               / "classes.txt").read_text().split("\n")
             if line.strip()]
    print(f"detector: {weights}\nclasses : {names}")

    rig = WebcamRig()
    rig.start()
    hands = HandIntruderDetector(list(rig.cameras))
    frames = {}
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        frames = {n: f for n, (f, _) in rig.frames().items()}
        if len(frames) >= len(rig.cameras):
            break
        time.sleep(0.1)
    if len(frames) < 2:
        rig.stop()
        raise SystemExit("need at least two cameras")
    crops = {c: workspace_crop(rig.cameras[c], frames[c].shape) for c in frames}

    smoother = pointing.PointingSmoother()
    window = "point to select  [SPACE] freeze  [S] save  [Q] quit"
    cv2.namedWindow(window)
    frozen = None
    try:
        while True:
            if frozen is None:
                frames = {n: f for n, (f, _) in rig.frames().items()}
            else:
                frames = frozen
            if len(frames) < 2:
                time.sleep(0.05)
                continue
            for cam, frame in frames.items():
                if cam not in crops:
                    crops[cam] = workspace_crop(rig.cameras[cam], frame.shape)

            per_cam = detect(model, frames, crops)
            located = locate_objects(rig, per_cam)

            hand = hands.detect(frames)
            pointers = {cam: data["pointers"][0]
                        for cam, data in hand.items() if data.get("pointers")}
            ray = pointing.pointer_ray(rig, pointers,
                                       require_gesture=not args.no_gesture)
            chosen, info = (None, {"scores": {}, "ambiguous": False,
                                   "runner_up": None})
            if ray is not None and located:
                chosen, info = pointing.select_pointed(
                    {cls: pos for cls, (pos, _) in located.items()},
                    ray[0], ray[1], max_angle_deg=args.max_angle)
            selected = smoother.update(chosen, time.monotonic())

            tiles = []
            for cam in sorted(rig.cameras):
                if cam not in frames:
                    tiles.append(np.zeros((QUAD[1], QUAD[0], 3), np.uint8))
                    continue
                view = cv2.resize(frames[cam], QUAD)
                sx = QUAD[0] / frames[cam].shape[1]
                sy = QUAD[1] / frames[cam].shape[0]
                for cls, (box, conf) in per_cam.get(cam, {}).items():
                    picked = cls == selected
                    colour = COLOR_PICK if picked else COLOR_IDLE
                    thickness = 4 if picked else 1
                    p1 = (int(box[0] * sx), int(box[1] * sy))
                    p2 = (int(box[2] * sx), int(box[3] * sy))
                    cv2.rectangle(view, p1, p2, colour, thickness)
                    tag = names[cls] if cls < len(names) else str(cls)
                    angle = info["scores"].get(cls, {}).get("angle_deg")
                    if angle is not None:
                        tag += f"  {angle:.0f}deg"
                    cv2.putText(view, tag, (p1[0], max(14, p1[1] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour,
                                2 if picked else 1, cv2.LINE_AA)
                if ray is not None:
                    # Draw the finger's line as it appears from this camera.
                    origin, direction = ray
                    line = [origin + direction * t for t in (0.0, 900.0)]
                    pixels = rig.cameras[cam].project(
                        [p / 1000.0 for p in line])
                    if all(p is not None for p in pixels):
                        cv2.line(view,
                                 (int(pixels[0][0] * sx), int(pixels[0][1] * sy)),
                                 (int(pixels[1][0] * sx), int(pixels[1][1] * sy)),
                                 COLOR_RAY, 2, cv2.LINE_AA)
                        cv2.circle(view, (int(pixels[0][0] * sx),
                                          int(pixels[0][1] * sy)), 6,
                                   COLOR_RAY, -1)
                cv2.putText(view, cam, (8, QUAD[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                            cv2.LINE_AA)
                tiles.append(view)
            while len(tiles) < 4:
                tiles.append(np.zeros((QUAD[1], QUAD[0], 3), np.uint8))

            panel = np.zeros((QUAD[1], QUAD[0], 3), np.uint8)
            if selected is not None:
                label = names[selected] if selected < len(names) else str(selected)
                cv2.putText(panel, label.upper(), (16, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_PICK, 3,
                            cv2.LINE_AA)
                cv2.putText(panel, "<- pointing at this", (16, 96),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PICK, 1,
                            cv2.LINE_AA)
            elif ray is None:
                cv2.putText(panel, "point with your index finger", (16, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1,
                            cv2.LINE_AA)
                cv2.putText(panel, "(two cameras must see the hand)", (16, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1,
                            cv2.LINE_AA)
            else:
                cv2.putText(panel, "nothing in the cone", (16, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1,
                            cv2.LINE_AA)
            if info.get("ambiguous"):
                other = info.get("runner_up")
                other = names[other] if (other is not None
                                         and other < len(names)) else "?"
                cv2.putText(panel, f"close call with {other} - move your hand",
                            (16, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            COLOR_AMBIG, 1, cv2.LINE_AA)

            row = 170
            cv2.putText(panel, "angle off your finger", (16, row - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                        cv2.LINE_AA)
            for cls, (position, cams) in sorted(located.items()):
                label = names[cls] if cls < len(names) else str(cls)
                score = info["scores"].get(cls)
                text = f"{label:<16}"
                text += (f"{score['angle_deg']:5.0f}deg  "
                         f"{score['forward_mm'] / 10:4.0f}cm away"
                         if score else "   (no ray)")
                colour = COLOR_PICK if cls == selected else (170, 170, 170)
                cv2.putText(panel, text, (16, row),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
                            cv2.LINE_AA)
                row += 26
            if not located:
                cv2.putText(panel, "no tool seen by two cameras", (16, row),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1,
                            cv2.LINE_AA)
            cv2.putText(panel,
                        ("FROZEN  " if frozen is not None else "")
                        + f"gesture gate {'off' if args.no_gesture else 'on'}"
                        + f"   cone {args.max_angle:.0f}deg",
                        (16, QUAD[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (140, 140, 140), 1, cv2.LINE_AA)
            tiles[3] = panel

            canvas = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == 32:
                frozen = None if frozen is not None else dict(frames)
            if key in (ord("s"), ord("S")):
                out = Path.home() / f"pointing_{int(time.time())}.jpg"
                cv2.imwrite(str(out), canvas)
                print(f"saved {out}")
    finally:
        cv2.destroyAllWindows()
        hands.close()
        rig.stop()


if __name__ == "__main__":
    main()
