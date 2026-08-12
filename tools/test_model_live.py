#!/usr/bin/env python3
"""Try a detector on the real cameras, with the robot left alone.

Runs exactly what the pick-and-place app runs - the same workspace crop and
upscale, the same confidence, the same ray-gap triangulation - so what you see
here is what the robot would act on. Nothing moves: this answers "would it
find the tool, and do the three views agree about where it is" before that
question is asked with a moving arm.

    python3 tools/test_model_live.py                          # newest run
    python3 tools/test_model_live.py --weights path/to.pt
    python3 tools/test_model_live.py --compare old.pt         # side by side

Keys: SPACE freeze | S save a frame | Q quit
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

from object_detection.webcam_rig import WebcamRig      # noqa: E402

# Same numbers the app uses - a test on different settings tests nothing.
CONF = 0.15
IMGSZ = 960
CROP_MARGIN_PX = 20
CROP_MAX_SCALE = 3.0
MAX_RAY_GAP_M = 0.040
TARGET_Z_MM = (-30.0, 120.0)
BOX_X, BOX_Y = (120.0, 780.0), (-450.0, 450.0)
WORKSPACE_CORNERS_M = [np.array([x, y, z])
                       for x in (0.12, 0.78) for y in (-0.45, 0.45)
                       for z in (0.0, 0.25)]
QUAD = (640, 360)
PALETTE = [(60, 60, 255), (200, 180, 0), (80, 220, 80), (255, 120, 255),
           (0, 200, 255), (255, 255, 255)]


def newest_weights():
    runs = sorted((Path.home() / "tool_dataset" / "runs").glob("*/weights/best.pt"),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("no trained weights under ~/tool_dataset/runs")
    return runs[-1]


def workspace_crop(camera, shape):
    """The projected workspace in this camera, as (x1, y1, x2, y2, scale)."""
    height, width = shape[:2]
    pixels = [p for p in camera.project(WORKSPACE_CORNERS_M) if p is not None]
    if len(pixels) < 4:
        return 0, 0, width, height, 1.0
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    x1 = int(max(0, min(xs) - CROP_MARGIN_PX))
    y1 = int(max(0, min(ys) - CROP_MARGIN_PX))
    x2 = int(min(width, max(xs) + CROP_MARGIN_PX))
    y2 = int(min(height, max(ys) + CROP_MARGIN_PX))
    if x2 - x1 < 40 or y2 - y1 < 40:
        return 0, 0, width, height, 1.0
    scale = min(CROP_MAX_SCALE, 1280.0 / max(1, x2 - x1))
    return x1, y1, x2, y2, scale


def detect(model, rig, frames, crops, names):
    """{cam: [ (cls, conf, box_full_frame) ]} using the app's crop+upscale."""
    cams = list(frames)
    images = []
    for cam in cams:
        x1, y1, x2, y2, scale = crops[cam]
        crop = frames[cam][y1:y2, x1:x2]
        if scale > 1.01:
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        images.append(crop)
    results = model(images, verbose=False, conf=CONF, imgsz=IMGSZ)
    out = {}
    for cam, result in zip(cams, results):
        x1, y1, _, _, scale = crops[cam]
        dets = []
        for box, conf, cls in zip(result.boxes.xyxy.tolist(),
                                  result.boxes.conf.tolist(),
                                  result.boxes.cls.tolist()):
            dets.append((int(cls), float(conf),
                         [box[0] / scale + x1, box[1] / scale + y1,
                          box[2] / scale + x1, box[3] / scale + y1]))
        out[cam] = dets
    return out


def triangulate_best(rig, per_cam, class_id):
    """Best 3D fix for one class across cameras, or None.

    Same gate as the app: two cameras minimum, rays must agree to 40 mm."""
    picks = {}
    for cam, dets in per_cam.items():
        same = [d for d in dets if d[0] == class_id]
        if same:
            best = max(same, key=lambda d: d[1])
            picks[cam] = ((best[2][0] + best[2][2]) / 2.0,
                          (best[2][1] + best[2][3]) / 2.0)
    if len(picks) < 2:
        return None, None, list(picks)
    point, gap = rig.triangulate(picks)
    return point, gap, list(picks)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--compare", default=None,
                        help="second model to run on the same frames")
    parser.add_argument("--classes", default=None,
                        help="classes.txt (default: the labelled dataset's)")
    args = parser.parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights) if args.weights else newest_weights()
    classes_path = Path(args.classes) if args.classes else (
        Path.home() / "tool_dataset" / "labeled" / "classes.txt")
    names = [line for line in classes_path.read_text().split("\n") if line.strip()]
    print(f"model   : {weights}")
    print(f"classes : {names}")

    model = YOLO(str(weights))
    model.to("cuda")
    other = None
    if args.compare:
        other = YOLO(args.compare)
        other.to("cuda")
        print(f"compare : {args.compare}")

    rig = WebcamRig()
    rig.start()
    print("waiting for camera frames...")
    # Cameras come up at their own pace - one C270 here regularly takes a
    # couple of seconds longer than the others. Wait for all of them, settle
    # for two, and let a straggler join later (crops are built on first sight).
    frames = {}
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        frames = {name: frame for name, (frame, _) in rig.frames().items()}
        if len(frames) >= len(rig.cameras):
            break
        time.sleep(0.1)
    if len(frames) < 2:
        rig.stop()
        raise SystemExit("fewer than two cameras delivered a frame - is the "
                         "robot app holding them?")
    crops = {}
    missing = sorted(set(rig.cameras) - set(frames))
    print(f"cameras : {', '.join(sorted(frames))}"
          + (f"   (waiting on {', '.join(missing)})" if missing else ""))

    window = "model test  [SPACE] freeze  [S] save  [Q] quit"
    cv2.namedWindow(window)
    # Everything below runs inside the try/finally that stops the rig: an
    # exception with capture threads still running aborts the process.
    frozen = None
    fps = 0.0
    try:
        while True:
            started = time.monotonic()
            if frozen is None:
                frames = {name: frame
                          for name, (frame, _) in rig.frames().items()}
            else:
                frames = frozen
            if len(frames) < 2:
                time.sleep(0.05)
                continue
            for cam, frame in frames.items():
                if cam not in crops:
                    crops[cam] = workspace_crop(rig.cameras[cam], frame.shape)
                    print(f"  {cam} online")

            per_cam = detect(model, rig, frames, crops, names)
            per_cam_old = (detect(other, rig, frames, crops, names)
                           if other is not None else {})

            # 3D agreement per class - the question the robot actually asks.
            fixes = {}
            for class_id in range(len(names)):
                point, gap, cams = triangulate_best(rig, per_cam, class_id)
                if point is not None and gap is not None:
                    fixes[class_id] = (point * 1000.0, gap * 1000.0, cams)

            tiles = []
            for cam in sorted(rig.cameras):
                if cam not in frames:
                    tiles.append(np.zeros((QUAD[1], QUAD[0], 3), np.uint8))
                    continue
                view = cv2.resize(frames[cam], QUAD)
                sx, sy = QUAD[0] / frames[cam].shape[1], QUAD[1] / frames[cam].shape[0]
                x1, y1, x2, y2, _ = crops[cam]
                cv2.rectangle(view, (int(x1 * sx), int(y1 * sy)),
                              (int(x2 * sx), int(y2 * sy)), (70, 70, 70), 1)
                for cls, conf, box in per_cam_old.get(cam, []):
                    cv2.rectangle(view, (int(box[0] * sx), int(box[1] * sy)),
                                  (int(box[2] * sx), int(box[3] * sy)),
                                  (120, 120, 120), 1)
                for cls, conf, box in per_cam.get(cam, []):
                    colour = PALETTE[cls % len(PALETTE)]
                    cv2.rectangle(view, (int(box[0] * sx), int(box[1] * sy)),
                                  (int(box[2] * sx), int(box[3] * sy)), colour, 2)
                    cv2.putText(view, f"{names[cls]} {conf:.2f}",
                                (int(box[0] * sx), max(14, int(box[1] * sy) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
                                cv2.LINE_AA)
                cv2.putText(view, cam, (8, QUAD[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                            cv2.LINE_AA)
                tiles.append(view)
            while len(tiles) < 4:
                tiles.append(np.zeros((QUAD[1], QUAD[0], 3), np.uint8))

            # Fourth tile: the 3D verdict, which is what the robot would use.
            panel = np.zeros((QUAD[1], QUAD[0], 3), np.uint8)
            cv2.putText(panel, "3D fix (>=2 cams, gap<40mm)", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)
            row = 60
            for class_id, name in enumerate(names):
                colour = PALETTE[class_id % len(PALETTE)]
                seen = sum(1 for dets in per_cam.values()
                           if any(d[0] == class_id for d in dets))
                if class_id in fixes:
                    point, gap, cams = fixes[class_id]
                    on_table = TARGET_Z_MM[0] <= point[2] <= TARGET_Z_MM[1]
                    ok = gap <= MAX_RAY_GAP_M * 1000
                    text = (f"{name:<16} {point[0]:6.0f},{point[1]:6.0f},"
                            f"{point[2]:5.0f}  gap {gap:4.0f}mm  {len(cams)}cam")
                    mark = "OK " if (ok and on_table) else ("gap" if not ok
                                                           else "z! ")
                    cv2.putText(panel, mark + text, (12, row),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                                colour if (ok and on_table) else (90, 90, 90),
                                1, cv2.LINE_AA)
                else:
                    cv2.putText(panel, f"--  {name:<16} seen in {seen} cam",
                                (12, row), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                                (90, 90, 90), 1, cv2.LINE_AA)
                row += 26
            cv2.putText(panel, f"{fps:.1f} fps"
                        + ("   FROZEN" if frozen is not None else ""),
                        (12, QUAD[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (160, 160, 160), 1, cv2.LINE_AA)
            if other is not None:
                cv2.putText(panel, "grey boxes = comparison model",
                            (12, QUAD[1] - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (140, 140, 140), 1, cv2.LINE_AA)
            tiles[3] = panel

            canvas = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
            cv2.imshow(window, canvas)
            fps = 0.8 * fps + 0.2 / max(1e-3, time.monotonic() - started)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == 32:
                frozen = None if frozen is not None else dict(frames)
            if key in (ord("s"), ord("S")):
                out = Path.home() / f"model_test_{int(time.time())}.jpg"
                cv2.imwrite(str(out), canvas)
                print(f"saved {out}")
    finally:
        cv2.destroyAllWindows()
        rig.stop()


if __name__ == "__main__":
    main()
