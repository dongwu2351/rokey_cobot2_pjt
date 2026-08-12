#!/usr/bin/env python3
"""Move the dataset to the four classes this cell actually contains.

The old five came from someone else's tool set. On this bench there are two
screwdrivers - one red/black, one teal - and telling them apart matters when
the robot is asked for a specific one, so they become separate classes. Drill
and pliers are not on the bench at all; every box carrying those labels is a
false positive and is dropped.

The red/teal split is done by colour, not by hand: the handles sit in two
clean hue clusters (red near 0/180, teal near 95), with only a few percent
in between. Anything ambiguous is left for review rather than guessed.

    python3 tools/remap_classes.py --dry-run     # what would change
    python3 tools/remap_classes.py               # do it (labels are backed up)

Known limitation, and the reason the changed images go back on the review
queue: the old detector confuses the blue WRENCH with a teal screwdriver, and
colour cannot separate those two - only shape can, which means you.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ROOT = Path.home() / "tool_dataset"

OLD_CLASSES = ["drill", "hammer", "pliers", "screwdriver", "wrench"]
NEW_CLASSES = ["screwdriver_red", "screwdriver", "wrench", "hammer"]

#: old index -> new index, or None to drop the box entirely.
DIRECT_MAP = {
    0: None,        # drill: not on this bench
    1: 3,           # hammer
    2: None,        # pliers: not on this bench
    4: 2,           # wrench
}
SPLIT_CLASS = 3     # old "screwdriver" -> decided by handle colour

RED_HUE = 15.0      # |hue| within this of 0/180 is the red handle
TEAL_LO, TEAL_HI = 75.0, 115.0
MIN_COLOURED_PIXELS = 20


def _circular_mean_hue(hues):
    """Circular because red straddles the 0/180 wrap - a plain average of
    hues 3 and 178 is 90, i.e. teal, which is the one answer that must never
    come out of this function."""
    angles = hues.astype(float) * 2.0 * np.pi / 180.0
    mean = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return (np.degrees(mean) / 2.0) % 180.0


def handle_hue(crop, relaxed=False):
    """Representative hue of a crop's coloured pixels, or None.

    Two passes, because a fixed saturation floor throws away every small or
    dimly lit box (100 of 1144 here). The strict pass decides most of them;
    the relaxed pass judges the rest against the crop's OWN saturation
    distribution, which recovers all but 29."""
    if crop.size == 0 or min(crop.shape[:2]) < 6:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
    if not relaxed:
        mask = (sat > 80) & (val > 60)
        if mask.sum() < MIN_COLOURED_PIXELS:
            return None
        return _circular_mean_hue(hue[mask])
    lit = val > 40
    if lit.sum() < 12:
        return None
    floor = max(45.0, float(np.percentile(sat[lit], 80)))
    mask = lit & (sat >= floor)
    if mask.sum() < 10:
        return None
    return _circular_mean_hue(hue[mask])


def _bucket(hue):
    if hue is None:
        return None
    if hue < RED_HUE or hue > 180.0 - RED_HUE:
        return 0
    if TEAL_LO < hue < TEAL_HI:
        return 1
    return None


def classify_screwdriver(crop):
    """(new class index, why) for an old 'screwdriver' box."""
    hue = handle_hue(crop)
    new = _bucket(hue)
    if new is not None:
        return new, f"{'red' if new == 0 else 'teal'} hue {hue:.0f}"
    hue = handle_hue(crop, relaxed=True)
    new = _bucket(hue)
    if new is not None:
        return new, f"{'red' if new == 0 else 'teal'} hue {hue:.0f} (relaxed)"
    return None, "colour unclear"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser() / "labeled"
    labels_dir = root / "labels"
    if not labels_dir.is_dir():
        raise SystemExit(f"no labels in {root}")

    stats = Counter()
    changed_images = []
    pending = []                       # (path, new_lines, new_sources)

    for label_path in sorted(labels_dir.glob("*.txt")):
        lines = [line.split() for line in label_path.read_text().split("\n")
                 if line.strip()]
        if not lines:
            continue
        source_path = root / "sources" / f"{label_path.stem}.json"
        sources = []
        if source_path.is_file():
            try:
                sources = json.loads(source_path.read_text())
            except Exception:
                sources = []

        image = None
        if any(int(float(parts[0])) == SPLIT_CLASS for parts in lines):
            image = cv2.imread(str(root / "images" / f"{label_path.stem}.jpg"))

        out_lines, out_sources, touched = [], [], False
        for index, parts in enumerate(lines):
            old = int(float(parts[0]))
            meta = sources[index] if index < len(sources) else {
                "source": "model", "conf": 1.0}
            if old == SPLIT_CLASS:
                new = None
                why = "no image"
                if image is not None:
                    height, width = image.shape[:2]
                    cx, cy, bw, bh = (float(v) for v in parts[1:])
                    x1 = max(0, int((cx - bw / 2) * width))
                    y1 = max(0, int((cy - bh / 2) * height))
                    x2 = min(width, int((cx + bw / 2) * width))
                    y2 = min(height, int((cy + bh / 2) * height))
                    new, why = classify_screwdriver(image[y1:y2, x1:x2])
                if new is None:
                    stats[f"screwdriver dropped ({why.split()[0]})"] += 1
                    touched = True
                    continue
                stats[f"screwdriver -> {NEW_CLASSES[new]}"] += 1
                out_lines.append(" ".join([str(new)] + parts[1:]))
                # A colour-split box is a guess about which screwdriver it is,
                # so it goes back on the review queue coloured as one.
                out_sources.append({"source": "geometry",
                                    "conf": float(meta.get("conf", 1.0))})
                touched = True
                continue
            new = DIRECT_MAP.get(old)
            if new is None:
                stats[f"{OLD_CLASSES[old]} dropped"] += 1
                touched = True
                continue
            if new != old:
                touched = True
            stats[f"{OLD_CLASSES[old]} -> {NEW_CLASSES[new]}"] += 1
            out_lines.append(" ".join([str(new)] + parts[1:]))
            out_sources.append(meta)

        if touched:
            changed_images.append(label_path.stem)
        pending.append((label_path, out_lines, out_sources))

    print("boxes")
    for key in sorted(stats):
        print(f"  {key:42s} {stats[key]:5d}")
    print(f"\nimages touched: {len(changed_images)} of {len(pending)}")
    print(f"classes: {OLD_CLASSES} -> {NEW_CLASSES}")

    if args.dry_run:
        print("\ndry run, nothing written")
        return

    backup = root.parent / "labeled_5class_backup"
    if not backup.exists():
        shutil.copytree(root, backup, dirs_exist_ok=False)
        print(f"\nbacked up to {backup}")

    for label_path, out_lines, out_sources in pending:
        label_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
        (root / "sources" / f"{label_path.stem}.json").write_text(
            json.dumps(out_sources))
    (root / "classes.txt").write_text("\n".join(NEW_CLASSES) + "\n")
    (root / "class_names.json").write_text(json.dumps(
        {str(i): name for i, name in enumerate(NEW_CLASSES)},
        ensure_ascii=False, indent=4))

    # Everything whose class was rewritten deserves another look, and the
    # colour split is exactly the kind of guess a human should confirm.
    queue = root / "review_first.txt"
    existing = []
    if queue.is_file():
        existing = [line for line in queue.read_text().split("\n") if line.strip()]
    already = {line.split()[0] for line in existing}
    added = [f"{stem}.jpg  class remap - check screwdriver vs wrench"
             for stem in changed_images if f"{stem}.jpg" not in already]
    queue.write_text("\n".join(added + existing) + "\n")
    reviewed = root / "reviewed.txt"
    if reviewed.is_file():
        keep = [line for line in reviewed.read_text().split("\n")
                if line.strip() and line.strip() not in set(changed_images)]
        reviewed.write_text("\n".join(keep) + "\n")

    print(f"review queue: +{len(added)} images -> {len(added) + len(existing)}")
    print("\nnext:  python3 tools/review_labels.py")


if __name__ == "__main__":
    main()
