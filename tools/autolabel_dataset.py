#!/usr/bin/env python3
"""Pre-label captured shots so a human only has to fix what is wrong.

Two passes, and the second one is the point of the calibrated rig:

1. the current detector labels every image (low confidence, so it over-calls
   rather than misses);
2. wherever TWO webcams agree on an object, its 3D position is triangulated
   and projected into the views that missed it - which produces correct boxes
   for exactly the far/oblique frames the detector is weakest on, i.e. the
   ones a human would otherwise have to draw by hand.

The output is plain YOLO txt, so any review tool takes it (X-AnyLabeling,
labelImg, CVAT, or a Roboflow upload). Nothing here is trusted blindly:
`review_first.txt` ranks the images by how likely they are to be wrong, so an
hour of review goes where it pays.

    python3 tools/autolabel_dataset.py                    # label
    python3 tools/autolabel_dataset.py --preview          # + drawn previews
    python3 tools/autolabel_dataset.py --split            # build train/val

Box colours in the previews: green = detector, cyan = filled in from the other
cameras (check these first), yellow = low confidence.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "pick_and_place_voice"))

DEFAULT_ROOT = Path.home() / "tool_dataset"
DEFAULT_WEIGHTS = (WORKSPACE / "pick_and_place_voice" / "resource"
                   / "yolov8n_tools_0122.pt")
DEFAULT_CLASSES = (WORKSPACE / "pick_and_place_voice" / "resource"
                   / "class_name_tool.json")

WEBCAMS = ("cam0", "cam1", "cam2")

try:                                        # ultralytics >= 8.4
    from ultralytics.cfg import DEFAULT_CFG_DICT as _CFG
    _HAS_QUANTIZE = "quantize" in _CFG
except Exception:
    _HAS_QUANTIZE = False
QUANTIZE_KEY = "quantize" if _HAS_QUANTIZE else "half"
QUANTIZE_VALUE = 16 if _HAS_QUANTIZE else True

#: Believe a detection enough to build 3D geometry from it.
CONF_TRUST = 0.35
#: Write it as a label, but send the image to the top of the review list.
CONF_KEEP = 0.15
#: Rays this far apart are not the same object (same gate as the live app).
MAX_RAY_GAP_M = 0.040
#: Refuse to invent a box smaller/larger than this fraction of the image.
MIN_BOX_FRAC, MAX_BOX_FRAC = 0.004, 0.6

#: Tools lie on the table, so a webcam box whose centre ray crosses the table
#: plane outside the workspace is looking at something else in the room - the
#: lab is full of stands and cables the detector calls "screwdriver". Height
#: of the table surface (m) and the workspace it is allowed to land in (mm).
TABLE_Z_M = 0.030
WORKSPACE_X_MM = (120.0 - 80.0, 780.0 + 80.0)
WORKSPACE_Y_MM = (-450.0 - 80.0, 450.0 + 80.0)


def load_classes(path):
    with open(path, "r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    return [mapping[key] for key in sorted(mapping, key=int)]


def parse_shot(stem):
    """'20260810_120000_00007_cam1' -> ('20260810_120000_00007', 'cam1')."""
    shot, _, cam = stem.rpartition("_")
    return shot, cam


def yolo_line(cls, box, width, height):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def link_or_copy(source: Path, target: Path):
    if target.exists():
        return
    try:
        os.link(source, target)                  # same disk: free, real file
    except OSError:
        shutil.copy2(source, target)


def read_labels(root: Path, stem: str, shape):
    """Existing YOLO labels as detections, marked as human-verified."""
    path = root / "labels" / f"{stem}.txt"
    if not path.is_file():
        return []
    height, width = shape[:2]
    out = []
    for line in path.read_text().split("\n"):
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = (float(v) for v in parts)
        out.append({"cls": int(cls), "conf": 1.0, "source": "human",
                    "box": [(cx - bw / 2) * width, (cy - bh / 2) * height,
                            (cx + bw / 2) * width, (cy + bh / 2) * height]})
    return out


def _centre(det):
    x1, y1, x2, y2 = det["box"]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _lands_on_table(rig, cam, det):
    """Does this box's centre ray cross the table inside the workspace?"""
    origin, direction = rig.cameras[cam].ray(*_centre(det))
    if abs(direction[2]) < 1e-6:
        return False
    distance = (TABLE_Z_M - origin[2]) / direction[2]
    if distance <= 0:
        return False
    point = (origin + distance * direction) * 1000.0
    return (WORKSPACE_X_MM[0] <= point[0] <= WORKSPACE_X_MM[1]
            and WORKSPACE_Y_MM[0] <= point[1] <= WORKSPACE_Y_MM[1])


def _seen_by_another_camera(rig, cam, det, per_cam):
    """True if a second webcam has a same-class box whose ray meets this one."""
    for other, dets in per_cam.items():
        if other == cam or other not in WEBCAMS:
            continue
        for candidate in dets:
            if candidate["cls"] != det["cls"]:
                continue
            point, gap = rig.triangulate({cam: _centre(det),
                                          other: _centre(candidate)})
            if point is not None and gap is not None and gap <= MAX_RAY_GAP_M:
                return True
    return False


def geometric_filter(rig, per_cam):
    """Split webcam detections into believable ones and suspects.

    The detector is generous at conf 0.15 - it calls a stand, a cable and a
    chair leg "screwdriver". Two facts the rig knows for free settle most of
    them: the tools are on the table, and a real object is visible to more
    than one camera. A box that fails both is set aside rather than deleted,
    because a genuinely single-view tool (occluded elsewhere) would fail the
    second test too - `suspects.txt` is there to get those back."""
    kept, suspects = {}, []
    for cam, dets in per_cam.items():
        if any(d["source"] == "human" for d in dets):
            kept[cam] = list(dets)               # already checked by a person
            continue
        if cam not in WEBCAMS:
            kept[cam] = list(dets)               # wrist camera moves; no gate
            continue
        keep = []
        for det in dets:
            if det["source"] == "geometry":
                keep.append(det)
                continue
            confirmed = _seen_by_another_camera(rig, cam, det, per_cam)
            on_table = _lands_on_table(rig, cam, det)
            if confirmed or (det["conf"] >= CONF_TRUST and on_table):
                det["confirmed"] = confirmed
                keep.append(det)
            else:
                suspects.append((cam, det, "off-table" if not on_table
                                 else "single view, low confidence"))
        kept[cam] = keep
    return kept, suspects


# ----------------------------------------------------------------------
# pass 2: fill the views that missed, using the cameras that did not
# ----------------------------------------------------------------------
def fill_missing_views(rig, per_cam, image_size):
    """per_cam: {cam: [det, ...]} -> [(cam, det), ...] newly invented.

    `det` is {"cls", "box", "conf", "source"}. One instance per class is
    assumed - two hammers in one shot would need matching across views, which
    is a different problem and is flagged for review instead."""
    invented = []
    classes = {det["cls"] for dets in per_cam.values() for det in dets}
    for cls in classes:
        trusted = {}
        for cam, dets in per_cam.items():
            if cam not in WEBCAMS:
                continue
            candidates = [d for d in dets
                          if d["cls"] == cls and d["conf"] >= CONF_TRUST]
            if len(candidates) == 1:
                trusted[cam] = candidates[0]
        if len(trusted) < 2:
            continue
        pixels = {}
        for cam, det in trusted.items():
            x1, y1, x2, y2 = det["box"]
            pixels[cam] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        point_m, gap = rig.triangulate(pixels)
        if point_m is None or gap is None or gap > MAX_RAY_GAP_M:
            continue
        # Physical size of the object, averaged over the views that saw it:
        # width_m = box_px * depth / fx.
        widths, heights = [], []
        for cam, det in trusted.items():
            camera = rig.cameras[cam]
            depth = camera.depth_of(point_m)
            if depth <= 0.05:
                continue
            x1, y1, x2, y2 = det["box"]
            widths.append((x2 - x1) * depth / camera.K[0, 0])
            heights.append((y2 - y1) * depth / camera.K[1, 1])
        if not widths:
            continue
        object_w, object_h = float(np.mean(widths)), float(np.mean(heights))

        for cam in WEBCAMS:
            if cam in trusted:
                continue
            if any(d["cls"] == cls for d in per_cam.get(cam, [])):
                continue                          # it did detect it, keep that
            camera = rig.cameras[cam]
            pixel = camera.project([point_m])[0]
            if pixel is None:
                continue
            depth = camera.depth_of(point_m)
            if depth <= 0.05:
                continue
            width_px = object_w * camera.K[0, 0] / depth
            height_px = object_h * camera.K[1, 1] / depth
            image_w, image_h = image_size
            if not (MIN_BOX_FRAC * image_w <= width_px <= MAX_BOX_FRAC * image_w):
                continue
            x1 = pixel[0] - width_px / 2.0
            y1 = pixel[1] - height_px / 2.0
            x2 = pixel[0] + width_px / 2.0
            y2 = pixel[1] + height_px / 2.0
            # A box mostly outside the frame is not a usable label.
            visible = (min(x2, image_w) - max(x1, 0.0)) * \
                      (min(y2, image_h) - max(y1, 0.0))
            if visible < 0.45 * width_px * height_px:
                continue
            box = [max(0.0, x1), max(0.0, y1),
                   min(image_w - 1.0, x2), min(image_h - 1.0, y2)]
            invented.append((cam, {"cls": cls, "box": box, "conf": 0.0,
                                   "source": "geometry"}))
    return invented


def run_label(args):
    from ultralytics import YOLO
    from object_detection.webcam_rig import WebcamRig

    root = Path(args.root).expanduser()
    images_dir = root / "raw" / "images"
    if not images_dir.is_dir():
        raise SystemExit(f"no captures in {images_dir} - run capture_dataset.py first")
    out = root / "labeled"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "sources").mkdir(parents=True, exist_ok=True)
    if args.preview:
        (out / "preview").mkdir(parents=True, exist_ok=True)

    local = out / "classes.txt"
    if local.is_file() and not args.reset_classes:
        names = [line for line in local.read_text().split("\n") if line.strip()]
    else:
        names = load_classes(args.classes)
        local.write_text("\n".join(names) + "\n")

    model = YOLO(args.weights)
    model.to(args.device)
    rig = WebcamRig()                    # calibration only; no camera opened

    # Human work is the scarce resource here: a re-label must never overwrite
    # an image someone already checked, or every training round would reset
    # the review queue to zero. Verified labels are also the best possible
    # input for the cross-view pass, so they are fed back in as trusted
    # detections rather than merely skipped.
    protected = set()
    reviewed_path = out / "reviewed.txt"
    if reviewed_path.is_file() and not args.redo_reviewed:
        protected = {line.strip() for line
                     in reviewed_path.read_text().split("\n") if line.strip()}
    if protected:
        print(f"{len(protected)} reviewed images are protected from overwrite")

    # Belt and braces: labels are a few hundred KB of text, and they hold
    # hours of somebody's attention. Snapshot them before writing anything,
    # so a mistake in this script is an inconvenience rather than a loss.
    if (out / "labels").is_dir() and any((out / "labels").glob("*.txt")):
        stamp = max([int(p.name.split("_")[-1])
                     for p in out.parent.glob("label_snapshots/snap_*")]
                    or [0]) + 1
        snapshot = out.parent / "label_snapshots" / f"snap_{stamp:03d}"
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in ("labels", "sources"):
            if (out / name).is_dir():
                shutil.copytree(out / name, snapshot / name, dirs_exist_ok=True)
        for name in ("reviewed.txt", "classes.txt", "review_first.txt"):
            if (out / name).is_file():
                shutil.copy2(out / name, snapshot / name)
        keep = sorted((out.parent / "label_snapshots").glob("snap_*"))[-10:]
        for old_snapshot in sorted((out.parent / "label_snapshots").glob("snap_*")):
            if old_snapshot not in keep:
                shutil.rmtree(old_snapshot, ignore_errors=True)
        print(f"labels snapshotted to {snapshot}")

    paths = sorted(images_dir.glob("*.jpg"))
    shots = defaultdict(dict)
    for path in paths:
        shot, cam = parse_shot(path.stem)
        shots[shot][cam] = path
    print(f"{len(paths)} images in {len(shots)} shots")

    stats = defaultdict(int)
    review = []
    suspects = []
    for index, (shot, by_cam) in enumerate(sorted(shots.items()), 1):
        cams = sorted(by_cam)
        frames = [cv2.imread(str(by_cam[cam])) for cam in cams]
        if any(frame is None for frame in frames):
            print(f"  {shot}: unreadable image, skipped")
            continue
        # ultralytics renamed half -> quantize=16 and warns on every call if
        # the old name is used, which buries the progress output. Send the
        # new name where it exists, the old one on older installs.
        extra = {QUANTIZE_KEY: QUANTIZE_VALUE} if args.half else {}
        results = model(frames, verbose=False, conf=CONF_KEEP,
                        imgsz=args.imgsz, **extra)

        per_cam = {}
        for cam, (result, frame) in zip(cams, zip(results, frames)):
            stem = f"{shot}_{cam}"
            if stem in protected:
                per_cam[cam] = read_labels(out, stem, frame.shape)
                stats["protected_boxes"] += len(per_cam[cam])
                continue
            dets = []
            for box, conf, cls in zip(result.boxes.xyxy.tolist(),
                                      result.boxes.conf.tolist(),
                                      result.boxes.cls.tolist()):
                dets.append({"cls": int(cls), "box": list(box),
                             "conf": float(conf), "source": "model"})
            per_cam[cam] = dets
            stats["model_boxes"] += len(dets)

        height, width = frames[0].shape[:2]
        if not args.no_filter:
            try:
                per_cam, dropped = geometric_filter(rig, per_cam)
                stats["suspects"] += len(dropped)
                for cam, det, why in dropped:
                    suspects.append(f"{shot}_{cam}.jpg  {names[det['cls']]}"
                                    f" {det['conf']:.2f}  {why}")
            except Exception as error:
                print(f"  {shot}: filter pass failed ({error})")
        if not args.no_geometry:
            try:
                for cam, det in fill_missing_views(rig, per_cam, (width, height)):
                    per_cam.setdefault(cam, []).append(det)
                    stats["geometry_boxes"] += 1
            except Exception as error:            # never lose a shot to this
                print(f"  {shot}: geometry pass failed ({error})")

        for cam, frame in zip(cams, frames):
            dets = per_cam.get(cam, [])
            image_h, image_w = frame.shape[:2]
            stem = f"{shot}_{cam}"
            link_or_copy(by_cam[cam], out / "images" / f"{stem}.jpg")
            if stem in protected:
                stats["images"] += 1
                stats["protected_images"] += 1
                continue                          # leave the human's file alone
            lines = [yolo_line(d["cls"], d["box"], image_w, image_h)
                     for d in dets]
            (out / "labels" / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
            # Where each box came from, in label order. review_labels.py needs
            # it to colour a guessed box differently from a detected one; the
            # YOLO txt itself has nowhere to say.
            (out / "sources" / f"{stem}.json").write_text(json.dumps(
                [{"source": d["source"], "conf": round(d["conf"], 3)}
                 for d in dets]))
            stats["images"] += 1

            # Review priority: invented boxes first (they are guesses from
            # geometry), then shaky detections, then empty frames.
            invented = sum(1 for d in dets if d["source"] == "geometry")
            weak = sum(1 for d in dets
                       if d["source"] == "model" and d["conf"] < CONF_TRUST)
            crowded = len(dets) > 3
            if invented:
                review.append((0, f"{stem}.jpg  filled-in x{invented}"))
            elif not dets:
                review.append((1, f"{stem}.jpg  nothing found"))
            elif weak or crowded:
                review.append((2, f"{stem}.jpg  weak x{weak}"
                                  + ("  crowded" if crowded else "")))

            if args.preview:
                canvas = frame.copy()
                for det in dets:
                    x1, y1, x2, y2 = (int(v) for v in det["box"])
                    if det["source"] == "geometry":
                        colour, tag = (255, 255, 0), "geom"
                    elif det["conf"] < CONF_TRUST:
                        colour, tag = (0, 255, 255), f"{det['conf']:.2f}"
                    else:
                        colour, tag = (80, 220, 80), f"{det['conf']:.2f}"
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(canvas, f"{names[det['cls']]} {tag}",
                                (x1, max(14, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
                                cv2.LINE_AA)
                cv2.imwrite(str(out / "preview" / f"{stem}.jpg"), canvas,
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
        if index % 25 == 0 or index == len(shots):
            print(f"  {index}/{len(shots)} shots")

    review.sort()
    (out / "review_first.txt").write_text(
        "\n".join(text for _, text in review) + "\n")
    (out / "suspects.txt").write_text("\n".join(suspects) + "\n")
    report = {
        "images": stats["images"],
        "shots": len(shots),
        "boxes_from_model": stats["model_boxes"],
        "boxes_from_geometry": stats["geometry_boxes"],
        "needs_review": len(review),
        "boxes_set_aside": stats["suspects"],
        "weights": str(args.weights),
        "classes": names,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))

    total = stats["model_boxes"] + stats["geometry_boxes"]
    print(f"\n{stats['images']} images labelled -> {out}")
    print(f"  {stats['model_boxes']} boxes from the detector")
    if stats["protected_images"]:
        print(f"  {stats['protected_images']} reviewed images kept as they are"
              f" ({stats['protected_boxes']} verified boxes, reused to place"
              " the cross-view ones)")
    print(f"  {stats['suspects']} false-looking boxes set aside"
          f" -> {out / 'suspects.txt'}")
    print(f"  {stats['geometry_boxes']} boxes filled in from the other cameras"
          f" ({100.0 * stats['geometry_boxes'] / max(1, total):.0f}% of all boxes,"
          " and the hard ones)")
    print(f"  {len(review)} of {stats['images']} images want a human look"
          f" ({100.0 * len(review) / max(1, stats['images']):.0f}%)"
          f" -> {out / 'review_first.txt'}")
    if args.preview:
        print(f"  previews: {out / 'preview'}")
    print(f"\nreview, then:  python3 tools/autolabel_dataset.py --split")


def run_split(args):
    """Build a YOLO dataset from the reviewed labels.

    Splits by SHOT: four views of one scene are near-duplicates, so letting
    them straddle train and val would report a validation score the model has
    not earned."""
    root = Path(args.root).expanduser()
    source = root / "labeled"
    if not (source / "images").is_dir():
        raise SystemExit(f"nothing labelled yet in {source}")
    # The labelled folder owns the taxonomy once remap_classes.py has run;
    # falling back to the app's json would write a data.yaml that disagrees
    # with the label files and train a model with shuffled class ids.
    local = source / "classes.txt"
    if local.is_file():
        names = [line for line in local.read_text().split("\n") if line.strip()]
    else:
        names = load_classes(args.classes)
    dataset = root / "dataset"
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in (source / "images").glob("*.jpg"))
    by_shot = defaultdict(list)
    for stem in stems:
        by_shot[parse_shot(stem)[0]].append(stem)
    shots = sorted(by_shot)
    random.Random(args.seed).shuffle(shots)
    cut = max(1, int(len(shots) * (1.0 - args.val_fraction)))
    assignment = {shot: ("train" if i < cut else "val")
                  for i, shot in enumerate(shots)}

    counts = defaultdict(int)
    empty = 0
    for shot, members in by_shot.items():
        split = assignment[shot]
        for stem in members:
            label = source / "labels" / f"{stem}.txt"
            if label.is_file() and not label.read_text().strip():
                empty += 1
                if not args.keep_empty:
                    continue                     # background image, opt-in
            link_or_copy(source / "images" / f"{stem}.jpg",
                         dataset / "images" / split / f"{stem}.jpg")
            if label.is_file():
                link_or_copy(label, dataset / "labels" / split / f"{stem}.txt")
            counts[split] += 1

    (dataset / "data.yaml").write_text(
        f"path: {dataset}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    print(f"train {counts['train']} images / val {counts['val']} images "
          f"({len(shots)} shots, split by shot)")
    if empty:
        print(f"  {empty} label-free images "
              f"{'kept as background' if args.keep_empty else 'skipped (--keep-empty to include)'}")
    print(f"  {dataset / 'data.yaml'}")
    print("\ntrain:  yolo detect train "
          f"data={dataset / 'data.yaml'} model=yolov8n.pt epochs=100 imgsz=960")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help="detector used for the first pass")
    parser.add_argument("--classes", default=str(DEFAULT_CLASSES))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--half", action="store_true",
                        help="fp16 inference (about 1.5x on this GPU)")
    parser.add_argument("--preview", action="store_true",
                        help="write images with the boxes drawn on")
    parser.add_argument("--no-geometry", action="store_true",
                        help="skip the cross-camera fill-in pass")
    parser.add_argument("--redo-reviewed", action="store_true",
                        help="re-label images already reviewed by hand "
                             "(their corrections are lost)")
    parser.add_argument("--reset-classes", action="store_true",
                        help="take the class list from --classes again, "
                             "discarding a remapped labeled/classes.txt")
    parser.add_argument("--no-filter", action="store_true",
                        help="keep every detection, including off-table ones")
    parser.add_argument("--split", action="store_true",
                        help="build dataset/ from the reviewed labels")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-empty", action="store_true",
                        help="include images with no objects as background")
    args = parser.parse_args()
    run_split(args) if args.split else run_label(args)


if __name__ == "__main__":
    main()
