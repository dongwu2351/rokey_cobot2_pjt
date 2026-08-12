#!/usr/bin/env python3
"""Four-camera synchronised capture for growing the tool-detection dataset.

Grabs the three calibrated C270s and the wrist D435i in one shot, so every
capture yields four views of the same scene from four very different angles
and distances - which is exactly the variety the detector is missing when it
loses a hammer across the table.

Two things make this worth more than pointing a phone at the bench:

* the webcams are CALIBRATED, so a later pass can triangulate an object seen
  in two views and generate labels for the views that missed it (see
  autolabel_dataset.py) - the boxes you never have to draw;
* the wrist pose is recorded with every shot, so the D435i frames can join
  that geometry instead of being an unrelated pile of images.

    python3 tools/capture_dataset.py --label hammer   # SPACE shoots now
    python3 tools/capture_dataset.py --delay 3        # countdown, shooting alone

Two people is the fast way: one moves the tool, one presses SPACE when the
scene looks right. Watch the preview before each press - a frame where a body
completely hides the object from ONE webcam is worse than no frame, because
the cross-view pass in autolabel_dataset.py will triangulate from the two
clear views and paint a box onto whatever is blocking the third. Partial
occlusion by a hand is fine and worth having; a fully blocked view is not.

Keys: SPACE shoot | B burst | A auto | U undo | L label | Q quit
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "pick_and_place_voice"))

from object_detection.webcam_rig import WebcamRig            # noqa: E402

DEFAULT_ROOT = Path.home() / "tool_dataset"
QUAD = (640, 360)
CAMS = ("cam0", "cam1", "cam2")
WRIST = "d435"

#: Reject a shot that looks like the one before it. Capturing 300 frames of a
#: motionless bench trains nothing and costs an hour of review; the threshold
#: is mean absolute difference on a thumbnail, so it tracks "did the scene
#: change", not sensor noise.
DUPLICATE_MAD = 3.0


class WristCamera:
    """D435i colour frames + the flange pose, via ROS. Optional."""

    def __init__(self):
        self.node = None
        self.ok = False
        self._executor = None
        self._thread = None
        self._tf_buffer = None

    def start(self):
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            import tf2_ros
            from object_detection.realsense import ImgNode
        except Exception as error:
            print(f"[wrist] ROS unavailable ({error}); capturing webcams only")
            return False
        try:
            import rclpy
            if not rclpy.ok():
                rclpy.init()
            self.node = ImgNode()
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer,
                                                          self.node)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self.node)
            self._thread = threading.Thread(target=self._executor.spin,
                                            daemon=True)
            self._thread.start()
        except Exception as error:
            print(f"[wrist] could not start ({error}); capturing webcams only")
            return False
        for _ in range(50):                      # up to 5 s for the first frame
            if self.node.get_color_frame() is not None:
                self.ok = True
                print("[wrist] D435i online")
                return True
            time.sleep(0.1)
        print("[wrist] no frames on the colour topic; capturing webcams only")
        return False

    def frame(self):
        if not self.ok:
            return None
        try:
            return self.node.get_color_frame()
        except Exception:
            return None

    def flange_pose(self):
        """base_link -> link_6 as (xyz mm, quaternion), or None.

        Read from TF rather than the robot API so this tool never competes
        with the pick-and-place app for the DRFL connection."""
        if self._tf_buffer is None:
            return None
        try:
            from rclpy.time import Time
            transform = self._tf_buffer.lookup_transform(
                "base_link", "link_6", Time())
        except Exception:
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return {
            "xyz_mm": [translation.x * 1000.0, translation.y * 1000.0,
                       translation.z * 1000.0],
            "quat_xyzw": [rotation.x, rotation.y, rotation.z, rotation.w],
        }

    def stop(self):
        if self._executor is not None:
            self._executor.shutdown()
        if self.node is not None:
            self.node.destroy_node()


class Session:
    def __init__(self, root: Path, label: str):
        self.root = root
        self.images = root / "raw" / "images"
        self.meta = root / "raw" / "meta"
        self.images.mkdir(parents=True, exist_ok=True)
        self.meta.mkdir(parents=True, exist_ok=True)
        self.name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.label = label
        self.shots = 0
        self.files = 0
        self.history = []                        # for undo
        self._last_thumb = {}

    def is_duplicate(self, frames):
        """True when nothing in any view has changed since the last shot."""
        changed = False
        for name, frame in frames.items():
            thumb = cv2.cvtColor(cv2.resize(frame, (64, 36)),
                                 cv2.COLOR_BGR2GRAY).astype(np.int16)
            previous = self._last_thumb.get(name)
            if previous is None or float(np.mean(np.abs(thumb - previous))) > DUPLICATE_MAD:
                changed = True
        return not changed

    def remember(self, frames):
        for name, frame in frames.items():
            self._last_thumb[name] = cv2.cvtColor(
                cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.int16)

    def save(self, frames, pose):
        shot = f"{self.name}_{self.shots:05d}"
        written = []
        for name, frame in frames.items():
            path = self.images / f"{shot}_{name}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written.append(path)
        record = {
            "shot": shot,
            "session": self.name,
            "captured_at": time.time(),
            "label_hint": self.label,
            "cameras": sorted(frames),
            "flange_pose": pose,
            "note": "webcams are calibrated (see webcam_calibration bundle); "
                    "the d435 view moves with flange_pose",
        }
        meta_path = self.meta / f"{shot}.json"
        meta_path.write_text(json.dumps(record, indent=2))
        written.append(meta_path)
        self.history.append(written)
        self.shots += 1
        self.files += len(frames)
        self.remember(frames)
        return shot

    def undo(self):
        if not self.history:
            return None
        for path in self.history.pop():
            try:
                path.unlink()
            except OSError:
                pass
        self.shots = max(0, self.shots - 1)
        self._last_thumb.clear()                 # next shot is never a dup
        return "undone"


def tile(frames, order, size=QUAD):
    cells = []
    for name in order:
        frame = frames.get(name)
        if frame is None:
            cell = np.zeros((size[1], size[0], 3), np.uint8)
            cv2.putText(cell, f"{name}: no signal", (12, size[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 200), 1)
        else:
            cell = cv2.resize(frame, size)
            cv2.putText(cell, name, (10, size[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
        cells.append(cell)
    top = np.hstack(cells[:2])
    bottom = np.hstack(cells[2:4])
    return np.vstack([top, bottom])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(DEFAULT_ROOT),
                        help=f"dataset root (default {DEFAULT_ROOT})")
    parser.add_argument("--delay", type=float, default=0.0, metavar="SEC",
                        help="countdown before each shot, for shooting alone "
                             "(default 0: SPACE fires immediately)")
    parser.add_argument("--burst", type=int, default=5, metavar="N",
                        help="shots taken by the B key, 0.6 s apart (default 5)")
    parser.add_argument("--auto", type=float, default=0.0, metavar="SEC",
                        help="unattended mode, a shot every SEC seconds - only "
                             "useful when nobody is in front of the cameras")
    parser.add_argument("--label", default="",
                        help="what is on the bench this session (recorded in "
                             "the metadata, e.g. hammer,wrench)")
    parser.add_argument("--no-wrist", action="store_true",
                        help="skip the D435i (no ROS needed)")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="save even when nothing in the scene changed")
    args = parser.parse_args()

    session = Session(Path(args.root).expanduser(), args.label)
    rig = WebcamRig()
    rig.start()
    print(f"[rig] webcams: {', '.join(sorted(rig.cameras))}")

    wrist = WristCamera()
    if not args.no_wrist:
        wrist.start()

    window = ("capture dataset  [SPACE] shoot  [B] burst  [A] auto  "
              "[U] undo  [L] label  [Q] quit")
    cv2.namedWindow(window)
    auto_interval = max(0.0, args.auto)
    auto_on = auto_interval > 0
    last_auto = 0.0
    flash_until = 0.0
    message = "SPACE to shoot"
    # Pending shots: (fire_at, remaining). A burst keeps firing after the
    # countdown so a slow rotation of the object yields several angles.
    pending_at = None
    pending_left = 0

    try:
        while True:
            frames = {name: frame for name, (frame, _) in rig.frames().items()}
            wrist_frame = wrist.frame()
            if wrist_frame is not None:
                frames[WRIST] = wrist_frame

            now = time.monotonic()
            shoot = False
            if auto_on and now - last_auto >= auto_interval:
                shoot = True
                last_auto = now
            if pending_at is not None and now >= pending_at:
                shoot = True
                pending_left -= 1
                pending_at = (now + 0.6) if pending_left > 0 else None

            view = tile(frames, list(CAMS) + [WRIST])
            header = np.zeros((34, view.shape[1], 3), np.uint8)
            status = (f"shots {session.shots}   files {session.files}   "
                      f"delay {args.delay:.0f}s   "
                      f"{'AUTO ' + str(auto_interval) + 's   ' if auto_on else ''}"
                      f"label '{session.label or '-'}'   {message}")
            cv2.putText(header, status, (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
            canvas = np.vstack([header, view])
            if pending_at is not None:
                left = pending_at - now
                if left > 0.05:
                    text = f"{left:.0f}" if left >= 1.0 else "•"
                    cv2.putText(canvas, text,
                                (canvas.shape[1] // 2 - 40, canvas.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 220, 255),
                                8, cv2.LINE_AA)
                    cv2.putText(canvas, "step out of the cameras' view",
                                (canvas.shape[1] // 2 - 210,
                                 canvas.shape[0] // 2 + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255),
                                2, cv2.LINE_AA)
            if now < flash_until:
                cv2.rectangle(canvas, (0, 0),
                              (canvas.shape[1] - 1, canvas.shape[0] - 1),
                              (80, 255, 80), 6)
            cv2.imshow(window, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == 32:
                if args.delay > 0:
                    pending_at, pending_left = now + args.delay, 1
                    message = f"shooting in {args.delay:.0f}s - step aside"
                else:
                    shoot = True
            elif key in (ord("b"), ord("B")):
                pending_at = now + max(args.delay, 0.0)
                pending_left = max(1, args.burst)
                message = f"burst of {pending_left} - move the object slowly"
            elif key in (ord("a"), ord("A")):
                if auto_interval <= 0:
                    auto_interval = 1.0
                auto_on = not auto_on
                last_auto = now
                message = f"auto {'on' if auto_on else 'off'}"
            elif key in (ord("u"), ord("U")):
                message = "undo: " + (session.undo() or "nothing to undo")
            elif key in (ord("l"), ord("L")):
                session.label = input("label for this session (e.g. hammer): ").strip()
                message = f"label = {session.label or '-'}"

            if not shoot:
                continue
            if len(frames) < 2:
                message = "not enough cameras"
                continue
            if not args.keep_duplicates and session.is_duplicate(frames):
                message = "skipped: scene unchanged - move something"
                continue
            shot = session.save(frames, wrist.flange_pose())
            flash_until = now + 0.12
            message = f"saved {shot} ({len(frames)} views)"
            print(f"  {message}")
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rig.stop()
        wrist.stop()
        print(f"\n{session.shots} shots, {session.files} images -> "
              f"{session.images}")
        if session.shots:
            print("next:  python3 tools/autolabel_dataset.py "
                  f"--root {session.root}")


if __name__ == "__main__":
    main()
