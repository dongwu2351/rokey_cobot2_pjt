#!/usr/bin/env python3
"""Has a camera moved since the calibration? Answer without a checkerboard.

Two independent checks, both live:

PAIRWISE RAY GAP (needs tools on the table, no robot)
    Two cameras looking at one object give two rays; how close they come to
    meeting is the gap. Compute it for all three pairs. If ONE camera has
    moved, the two pairs containing it disagree while the pair that excludes
    it still meets cleanly - which names the culprit outright. Verified in
    simulation: rotating one camera by 0.3 deg leaves its own pair-free gap
    unchanged while both of its pairs blow up.

    (Leave-one-out reprojection is also reported, but it CANNOT isolate the
    culprit on its own: a bad camera poisons the triangulation the other two
    are judged against, so every camera looks bad at once.)

TCP CROSSHAIR (needs the robot driver, no tools)
    The flange pose is known exactly from the robot's own encoders and TF.
    Projecting it into each view puts a crosshair where the gripper must
    appear. Nothing about the cameras is involved in computing it, so any
    offset you see is calibration error, measured against ground truth.

    python3 tools/check_calibration.py                # both, 20 s of samples
    python3 tools/check_calibration.py --seconds 60   # more samples
    python3 tools/check_calibration.py --no-robot     # skip TF/crosshair

Keys: SPACE pause | S save the view | Q quit and print the verdict
"""

from __future__ import annotations

import argparse
import sys
import time
import itertools
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "pick_and_place_voice"))

from object_detection.webcam_rig import WebcamRig      # noqa: E402

CONF = 0.25
IMGSZ = 960
CROP_MARGIN_PX = 20
CROP_MAX_SCALE = 3.0
WORKSPACE_CORNERS_M = [np.array([x, y, z])
                       for x in (0.12, 0.78) for y in (-0.45, 0.45)
                       for z in (0.0, 0.25)]
QUAD = (640, 360)

#: Detection box centres do not mark the same physical point from different
#: angles, so a healthy rig still shows a few pixels of disagreement. These
#: thresholds separate "that floor" from "a camera has moved".
GOOD_PX, SUSPECT_PX = 12.0, 25.0


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
            centre = ((box[0] + box[2]) / 2.0 / scale + x1,
                      (box[1] + box[3]) / 2.0 / scale + y1)
            if cls not in best or conf > best[cls][1]:
                best[cls] = (centre, float(conf))
        out[cam] = best
    return out


class Tf:
    """base_link -> link_6 from TF, or nothing if ROS is not up."""

    def __init__(self):
        self.ok = False
        self._buffer = None
        try:
            import rclpy
            from rclpy.node import Node
            import tf2_ros
            import threading
            if not rclpy.ok():
                rclpy.init()
            self._node = Node("calibration_check_tf")
            self._buffer = tf2_ros.Buffer()
            self._listener = tf2_ros.TransformListener(self._buffer, self._node)
            self._thread = threading.Thread(
                target=lambda: rclpy.spin(self._node), daemon=True)
            self._thread.start()
            time.sleep(1.5)
            self.ok = self.position() is not None
        except Exception as error:
            print(f"[tf] not available ({error}) - skipping the TCP crosshair")

    def shutdown(self):
        try:
            import rclpy
            if self._buffer is not None and rclpy.ok():
                rclpy.shutdown()          # stop the spin thread before exit
        except Exception:
            pass

    def position(self):
        if self._buffer is None:
            return None
        try:
            from rclpy.time import Time
            t = self._buffer.lookup_transform("base_link", "link_6", Time())
        except Exception:
            return None
        v = t.transform.translation
        return np.array([v.x, v.y, v.z])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--seconds", type=float, default=20.0,
                        help="how long to gather samples before the verdict")
    parser.add_argument("--no-robot", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO
    weights = args.weights or sorted(
        (Path.home() / "tool_dataset" / "runs").glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime)[-1]
    model = YOLO(str(weights))
    model.to("cuda")
    print(f"detector: {weights}")

    rig = WebcamRig()
    rig.start()
    tf = Tf() if not args.no_robot else None

    frames = {}
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        frames = {n: f for n, (f, _) in rig.frames().items()}
        if len(frames) >= len(rig.cameras):
            break
        time.sleep(0.1)
    if len(frames) < 3:
        rig.stop()
        raise SystemExit(f"need all three cameras, got {sorted(frames)}")
    crops = {c: workspace_crop(rig.cameras[c], frames[c].shape) for c in frames}

    gaps = defaultdict(list)        # (cam_a, cam_b) -> [gap mm]
    errors = defaultdict(list)      # cam -> [reprojection px]
    depths = defaultdict(list)      # cam -> [depth m] (px -> mm conversion)
    heights = []                    # triangulated z of table objects (mm)
    window = "calibration check  [SPACE] pause  [S] save  [Q] verdict"
    cv2.namedWindow(window)
    started = time.monotonic()
    paused = False

    try:
        while True:
            if not paused:
                frames = {n: f for n, (f, _) in rig.frames().items()}
            if len(frames) < 3:
                time.sleep(0.05)
                continue
            per_cam = detect(model, frames, crops)

            # Leave-one-out: every class all three cameras agree they can see.
            shared = set.intersection(*[set(v) for v in per_cam.values()])
            for cls in shared:
                # Pairwise gaps: the discriminator. The pair WITHOUT the moved
                # camera keeps meeting cleanly, so a low gap names the innocents.
                for a, b in itertools.combinations(sorted(per_cam), 2):
                    point, gap = rig.triangulate({a: per_cam[a][cls][0],
                                                  b: per_cam[b][cls][0]})
                    if gap is not None:
                        gaps[(a, b)].append(gap * 1000.0)
                for held_out in per_cam:
                    others = {c: per_cam[c][cls][0]
                              for c in per_cam if c != held_out}
                    point, gap = rig.triangulate(others)
                    if point is None or gap is None or gap > 0.060:
                        continue
                    predicted = rig.cameras[held_out].project([point])[0]
                    if predicted is None:
                        continue
                    seen = per_cam[held_out][cls][0]
                    errors[held_out].append(float(np.hypot(
                        predicted[0] - seen[0], predicted[1] - seen[1])))
                    depths[held_out].append(
                        rig.cameras[held_out].depth_of(point))
                if len(per_cam) >= 3:
                    point, gap = rig.triangulate(
                        {c: per_cam[c][cls][0] for c in per_cam})
                    if point is not None:
                        heights.append(point[2] * 1000.0)

            tcp = tf.position() if (tf is not None and tf.ok) else None
            tiles = []
            for cam in sorted(rig.cameras):
                view = cv2.resize(frames[cam], QUAD)
                sx = QUAD[0] / frames[cam].shape[1]
                sy = QUAD[1] / frames[cam].shape[0]
                for cls, (centre, conf) in per_cam.get(cam, {}).items():
                    cv2.drawMarker(view, (int(centre[0] * sx), int(centre[1] * sy)),
                                   (80, 220, 80), cv2.MARKER_CROSS, 12, 2)
                if tcp is not None:
                    pixel = rig.cameras[cam].project([tcp])[0]
                    if pixel is not None:
                        px, py = int(pixel[0] * sx), int(pixel[1] * sy)
                        cv2.drawMarker(view, (px, py), (0, 200, 255),
                                       cv2.MARKER_TILTED_CROSS, 22, 2)
                        cv2.circle(view, (px, py), 16, (0, 200, 255), 1)
                        cv2.putText(view, "TCP", (px + 20, py),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 200, 255), 1, cv2.LINE_AA)
                samples = errors[cam]
                label = (f"{cam}   n={len(samples)}"
                         + (f"  med {np.median(samples):.1f}px"
                            if samples else "  gathering..."))
                cv2.putText(view, label, (8, QUAD[1] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                            cv2.LINE_AA)
                tiles.append(view)

            panel = np.zeros((QUAD[1], QUAD[0], 3), np.uint8)
            elapsed = time.monotonic() - started
            row = 26
            cv2.putText(panel, "pairwise ray gap (the discriminator)", (12, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
            row += 28
            for pair in sorted(gaps):
                values = gaps[pair]
                if len(values) < 5:
                    continue
                med = float(np.median(values))
                colour = ((80, 220, 80) if med < 15 else
                          (0, 200, 255) if med < 30 else (60, 60, 255))
                cv2.putText(panel, f"{pair[0]}-{pair[1]}   {med:5.1f} mm"
                            f"   n={len(values)}", (12, row),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
                row += 26
            row += 10
            cv2.putText(panel, "leave-one-out reprojection", (12, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
            row += 26
            for cam in sorted(rig.cameras):
                samples = errors[cam]
                if len(samples) < 10:
                    cv2.putText(panel, f"{cam}   n={len(samples)} (need 10+)",
                                (12, row), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (140, 140, 140), 1, cv2.LINE_AA)
                else:
                    med = float(np.median(samples))
                    mm = med * float(np.median(depths[cam])) * 1000.0 / \
                        rig.cameras[cam].K[0, 0]
                    colour = ((80, 220, 80) if med < GOOD_PX else
                              (0, 200, 255) if med < SUSPECT_PX else (60, 60, 255))
                    cv2.putText(panel,
                                f"{cam}   {med:5.1f} px  ~{mm:4.0f} mm   n={len(samples)}",
                                (12, row), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                colour, 1, cv2.LINE_AA)
                row += 30
            if heights:
                cv2.putText(panel,
                            f"table z: {np.median(heights):.0f} mm (expect 0..60)",
                            (12, row + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                            cv2.LINE_AA)
            cv2.putText(panel, f"{elapsed:.0f}s / {args.seconds:.0f}s"
                        + ("   PAUSED" if paused else "")
                        + ("   TCP: on" if tcp is not None else "   TCP: off"),
                        (12, QUAD[1] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (160, 160, 160), 1, cv2.LINE_AA)
            tiles.append(panel)

            canvas = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == 32:
                paused = not paused
            if key in (ord("s"), ord("S")):
                out = Path.home() / f"calibration_check_{int(time.time())}.jpg"
                cv2.imwrite(str(out), canvas)
                print(f"saved {out}")
            if elapsed > args.seconds and all(len(v) >= 20
                                              for v in gaps.values()) and gaps:
                break
    finally:
        cv2.destroyAllWindows()
        rig.stop()
        if tf is not None:
            tf.shutdown()

    print("\n=== leave-one-out reprojection ===")
    verdicts = {}
    for cam in sorted(errors):
        samples = np.array(errors[cam])
        if len(samples) < 10:
            print(f"  {cam}: only {len(samples)} samples - move a tool around "
                  "in view of all three cameras and run again")
            continue
        med = float(np.median(samples))
        p90 = float(np.percentile(samples, 90))
        mm = med * float(np.median(depths[cam])) * 1000.0 / rig.cameras[cam].K[0, 0]
        verdicts[cam] = med
        print(f"  {cam}: median {med:5.1f} px  p90 {p90:5.1f} px"
              f"  ~{mm:4.0f} mm at {np.median(depths[cam]):.2f} m"
              f"   (n={len(samples)})")
    print("\n=== pairwise ray gap (who moved) ===")
    medians = {pair: float(np.median(v)) for pair, v in gaps.items() if len(v) >= 10}
    for pair in sorted(medians):
        print(f"  {pair[0]}-{pair[1]}: median {medians[pair]:5.1f} mm"
              f"   (n={len(gaps[pair])})")
    if len(medians) == 3:
        cams = sorted(rig.cameras)
        # Each camera's score: the best gap among the pairs that EXCLUDE it.
        # For the moved camera that pair is the two good ones - small. For a
        # good camera every excluding pair contains the moved one - large.
        excluded = {c: min(v for pair, v in medians.items() if c not in pair)
                    for c in cams}
        suspect = min(excluded, key=excluded.get)
        others = sorted(v for c, v in excluded.items() if c != suspect)
        print()
        # A pair's gap also depends on its baseline, and box centres are not
        # the same physical point seen from two angles - so demanding the
        # innocent pair be twice as good was too strict to ever fire. What
        # matters is that ONE camera's excluding pair is clearly better AND
        # the difference is bigger than the few millimetres of box-centre
        # disagreement a healthy rig shows.
        margin = others[0] - excluded[suspect] if others else 0.0
        if max(medians.values()) < 12.0:
            print("  VERDICT: every pair agrees to within 12 mm - the rig looks "
                  "intact. Nothing to re-calibrate.")
        elif others and excluded[suspect] < 0.7 * others[0] and margin > 4.0:
            print(f"  VERDICT: {suspect} is the odd one out. The pair that "
                  f"excludes it meets to {excluded[suspect]:.1f} mm while every "
                  f"pair containing it is {others[0]:.1f} mm or worse - that is "
                  f"the signature of {suspect} having moved.")
            print(f"           Confirm with the TCP crosshair before "
                  f"re-calibrating: box centres are not the same physical "
                  f"point from two angles, so a few mm of this is expected "
                  f"even on a perfect rig.")
        else:
            print("  VERDICT: no single camera stands out - all pairs disagree "
                  "similarly. That points at something shared (the robot base "
                  "moved relative to the table, or the table itself), not one "
                  "camera.")
    if heights:
        print(f"\n  table objects triangulate to z = "
              f"{np.median(heights):.0f} mm (expect roughly 0..60 for tools "
              f"lying on the table)")
    print("\n  TCP crosshair: if the orange cross did not sit on the gripper "
          "in some view, that view's extrinsic is off.")


if __name__ == "__main__":
    main()
