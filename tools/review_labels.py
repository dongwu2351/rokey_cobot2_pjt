#!/usr/bin/env python3
"""Fix the auto-labels by hand, in the order that matters.

A general labelling tool starts you at image 1 of 1016 with no opinion about
which ones are wrong. This one walks `review_first.txt` - guessed boxes first,
then empty frames, then shaky detections - and leaves the ~700 images the
detector clearly got right alone.

    python3 tools/review_labels.py              # the review queue
    python3 tools/review_labels.py --all        # every image
    python3 tools/review_labels.py --class hammer   # only frames with hammers

Mouse:  drag on empty space = new box (gets the current class)
        click a box = select it     drag its corner = resize
        RIGHT-CLICK a box = delete it on the spot
Keys:   D / SPACE next     A previous     1-9 class of the selected box
        (no selection: 1-9 sets the class new boxes get)
        DEL / BACKSPACE / X delete selected     C clear every box here
        G accept a guessed box     U undo
        F flag for later     S save now     Q quit (saves)

Edits are written straight back into labeled/labels/*.txt, so training picks
them up with no export step. Progress lives in labeled/reviewed.txt: quit and
come back whenever.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ROOT = Path.home() / "tool_dataset"

COLOR_MODEL = (80, 220, 80)          # detector said so
COLOR_GUESS = (255, 255, 0)          # invented from the other cameras
COLOR_WEAK = (0, 200, 255)           # detector was unsure
COLOR_SELECT = (255, 255, 255)
HANDLE = 10                          # px grab radius for resizing
WEAK_CONF = 0.35


class Item:
    """One image: its boxes, and where they came from."""

    def __init__(self, root: Path, stem: str, reason: str = ""):
        self.stem = stem
        self.reason = reason
        self.image_path = root / "images" / f"{stem}.jpg"
        self.label_path = root / "labels" / f"{stem}.txt"
        self.source_path = root / "sources" / f"{stem}.json"
        self.boxes = []              # {cls, xyxy (px), source, conf}
        self.dirty = False
        self.image = None

    def load(self):
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            return False
        height, width = self.image.shape[:2]
        sources = []
        if self.source_path.is_file():
            try:
                sources = json.loads(self.source_path.read_text())
            except Exception:
                sources = []
        self.boxes = []
        if self.label_path.is_file():
            for index, line in enumerate(self.label_path.read_text().split("\n")):
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = (float(v) for v in parts)
                meta = sources[index] if index < len(sources) else {}
                self.boxes.append({
                    "cls": int(cls),
                    "xyxy": [(cx - bw / 2) * width, (cy - bh / 2) * height,
                             (cx + bw / 2) * width, (cy + bh / 2) * height],
                    "source": meta.get("source", "model"),
                    "conf": float(meta.get("conf", 1.0)),
                })
        self.dirty = False
        return True

    def save(self):
        if not self.dirty or self.image is None:
            return False
        height, width = self.image.shape[:2]
        lines, sources = [], []
        for box in self.boxes:
            x1, y1, x2, y2 = box["xyxy"]
            x1, x2 = sorted((max(0.0, x1), min(width - 1.0, x2)))
            y1, y2 = sorted((max(0.0, y1), min(height - 1.0, y2)))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue                      # a stray click is not a label
            lines.append(f"{box['cls']} {(x1 + x2) / 2 / width:.6f} "
                         f"{(y1 + y2) / 2 / height:.6f} "
                         f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
            sources.append({"source": box["source"],
                            "conf": round(box["conf"], 3)})
        self.label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        self.source_path.parent.mkdir(parents=True, exist_ok=True)
        self.source_path.write_text(json.dumps(sources))
        self.dirty = False
        return True


class Reviewer:
    def __init__(self, root: Path, items, names, scale_to=1600):
        self.root = root
        self.items = items
        self.names = names
        self.index = 0
        self.current_class = 1 if len(names) > 1 else 0
        self.selected = None
        self.drag_from = None
        self.drag_box = None
        self.drag_handle = None
        self.undo_stack = []
        self.scale = 1.0
        self.scale_to = scale_to
        self.reviewed = set()
        self.flagged = []
        self.reviewed_path = root / "reviewed.txt"
        if self.reviewed_path.is_file():
            self.reviewed = {line.strip() for line
                             in self.reviewed_path.read_text().split("\n")
                             if line.strip()}
        self.item = None
        self.window = "review labels"

    # -- helpers ------------------------------------------------------
    def to_image(self, x, y):
        return x / self.scale, y / self.scale

    def push_undo(self):
        self.undo_stack.append(json.dumps(self.item.boxes))
        del self.undo_stack[:-30]

    def undo(self):
        if not self.undo_stack:
            return
        self.item.boxes = json.loads(self.undo_stack.pop())
        self.item.dirty = True
        self.selected = None

    def box_at(self, x, y):
        """Topmost box containing the point, smallest first so a small box
        inside a big one is still reachable."""
        hits = [i for i, b in enumerate(self.item.boxes)
                if b["xyxy"][0] <= x <= b["xyxy"][2]
                and b["xyxy"][1] <= y <= b["xyxy"][3]]
        if not hits:
            return None
        return min(hits, key=lambda i: ((self.item.boxes[i]["xyxy"][2]
                                         - self.item.boxes[i]["xyxy"][0])
                                        * (self.item.boxes[i]["xyxy"][3]
                                           - self.item.boxes[i]["xyxy"][1])))

    def handle_at(self, index, x, y):
        """Which corner of box `index` is under the cursor, if any."""
        x1, y1, x2, y2 = self.item.boxes[index]["xyxy"]
        radius = HANDLE / self.scale
        for name, (hx, hy) in (("tl", (x1, y1)), ("tr", (x2, y1)),
                               ("bl", (x1, y2)), ("br", (x2, y2))):
            if abs(x - hx) <= radius and abs(y - hy) <= radius:
                return name
        return None

    # -- navigation ---------------------------------------------------
    def load(self, index):
        if self.item is not None and self.item.save():
            print(f"  saved {self.item.stem}")
        self.index = max(0, min(len(self.items) - 1, index))
        self.item = self.items[self.index]
        while not self.item.load():
            print(f"  unreadable: {self.item.stem}")
            if self.index >= len(self.items) - 1:
                return False
            self.index += 1
            self.item = self.items[self.index]
        self.selected = None
        self.undo_stack.clear()
        self.scale = min(1.0, self.scale_to / self.item.image.shape[1])
        return True

    def mark_reviewed(self):
        """Record progress on disk immediately, not at quit.

        A re-label pass protects whatever reviewed.txt lists at the moment it
        runs. Buffering progress until Q meant an hour of review could sit
        unlisted and be overwritten by the next training round - which is
        exactly what happened once. Writing a short text file per image costs
        nothing next to the work it protects."""
        if self.item.stem in self.reviewed:
            return
        self.reviewed.add(self.item.stem)
        try:
            with open(self.reviewed_path, "a", encoding="utf-8") as handle:
                handle.write(f"{self.item.stem}\n")
        except OSError as error:
            print(f"  could not record progress: {error}")

    def flush(self):
        if self.item is not None:
            self.item.save()
        # Rewrite once at the end to drop duplicates from the append log.
        self.reviewed_path.write_text("\n".join(sorted(self.reviewed)) + "\n")
        if self.flagged:
            (self.root / "flagged.txt").write_text("\n".join(self.flagged) + "\n")

    # -- mouse --------------------------------------------------------
    def on_mouse(self, event, x, y, flags, param):
        ix, iy = self.to_image(x, y)
        if event == cv2.EVENT_RBUTTONDOWN:
            # Right-click deletes: most of the review is removing boxes the
            # detector invented, and select-then-reach-for-a-key doubles the
            # work for the commonest action.
            hit = self.box_at(ix, iy)
            if hit is not None:
                self.push_undo()
                removed = self.item.boxes.pop(hit)
                self.selected = None
                self.item.dirty = True
                print(f"  deleted {self.names[removed['cls']]}")
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = self.box_at(ix, iy)
            if hit is not None:
                handle = self.handle_at(hit, ix, iy)
                self.selected = hit
                if handle:
                    self.push_undo()
                    self.drag_handle = (hit, handle)
                return
            self.selected = None
            self.drag_from = (ix, iy)
            self.drag_box = [ix, iy, ix, iy]
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_handle is not None:
                index, corner = self.drag_handle
                box = self.item.boxes[index]["xyxy"]
                if "l" in corner[1:] or corner in ("tl", "bl"):
                    box[0] = ix
                else:
                    box[2] = ix
                box[1 if corner[0] == "t" else 3] = iy
                self.item.dirty = True
            elif self.drag_from is not None:
                self.drag_box[2:] = [ix, iy]
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag_handle is not None:
                self.drag_handle = None
                return
            if self.drag_from is None:
                return
            x1, y1, x2, y2 = self.drag_box
            self.drag_from = self.drag_box = None
            if abs(x2 - x1) < 6 or abs(y2 - y1) < 6:
                return                          # a click, not a box
            self.push_undo()
            self.item.boxes.append({
                "cls": self.current_class,
                "xyxy": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
                "source": "human", "conf": 1.0,
            })
            self.selected = len(self.item.boxes) - 1
            self.item.dirty = True

    # -- drawing ------------------------------------------------------
    def render(self):
        canvas = cv2.resize(self.item.image, None, fx=self.scale, fy=self.scale)
        for index, box in enumerate(self.item.boxes):
            x1, y1, x2, y2 = (int(v * self.scale) for v in box["xyxy"])
            if box["source"] == "geometry":
                colour = COLOR_GUESS
            elif box["source"] == "human":
                colour = COLOR_MODEL
            elif box["conf"] < WEAK_CONF:
                colour = COLOR_WEAK
            else:
                colour = COLOR_MODEL
            thickness = 3 if index == self.selected else 2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)
            if index == self.selected:
                cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                              COLOR_SELECT, 1)
                for hx, hy in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                    cv2.rectangle(canvas, (hx - 4, hy - 4), (hx + 4, hy + 4),
                                  COLOR_SELECT, -1)
            tag = self.names[box["cls"]]
            if box["source"] == "geometry":
                tag += " GUESS"
            elif box["source"] == "model" and box["conf"] < WEAK_CONF:
                tag += f" {box['conf']:.2f}"
            cv2.putText(canvas, tag, (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
        if self.drag_box is not None:
            x1, y1, x2, y2 = (int(v * self.scale) for v in self.drag_box)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_SELECT, 1)

        bar = np.zeros((66, canvas.shape[1], 3), np.uint8)
        done = len(self.reviewed)
        cv2.putText(bar, f"{self.index + 1}/{len(self.items)}   "
                         f"reviewed {done}   {self.item.stem}"
                         f"   [{self.item.reason}]",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        classes = "  ".join(
            f"{i + 1}:{name}" + ("*" if i == self.current_class else "")
            for i, name in enumerate(self.names))
        cv2.putText(bar, f"new box class -> {classes}", (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 220, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(bar, "drag=new  L-click=select  R-click/X=delete  "
                         "C=clear all  G=accept guess  U=undo  F=flag  "
                         "A/D=prev/next  Q=quit",
                    (10, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170),
                    1, cv2.LINE_AA)
        cv2.imshow(self.window, np.vstack([bar, canvas]))

    # -- loop ---------------------------------------------------------
    def run(self):
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._mouse_with_offset)
        if not self.load(0):
            print("nothing to review")
            return
        while True:
            self.render()
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key in (ord("q"), ord("Q"), 27):
                self.mark_reviewed()
                break
            if key in (ord("d"), ord("D"), 32, 83):
                self.mark_reviewed()
                if self.index >= len(self.items) - 1:
                    print("end of queue")
                    break
                self.load(self.index + 1)
            elif key in (ord("a"), ord("A"), 81):
                self.load(self.index - 1)
            elif key in (8, 127, ord("x"), ord("X")):
                if self.selected is not None:
                    self.push_undo()
                    removed = self.item.boxes.pop(self.selected)
                    self.selected = None
                    self.item.dirty = True
                    print(f"  deleted {self.names[removed['cls']]}")
            elif key in (ord("c"), ord("C")):
                # Whole-frame wipe: a shot where the detector hallucinated
                # everything is faster to empty than to pick apart.
                if self.item.boxes:
                    self.push_undo()
                    print(f"  cleared {len(self.item.boxes)} boxes")
                    self.item.boxes = []
                    self.selected = None
                    self.item.dirty = True
            elif key in (ord("u"), ord("U")):
                self.undo()
            elif key in (ord("g"), ord("G")):
                # Accept a guessed box: it becomes an ordinary label, so the
                # next pass stops colouring it as unverified.
                if self.selected is not None:
                    self.push_undo()
                    self.item.boxes[self.selected]["source"] = "human"
                    self.item.boxes[self.selected]["conf"] = 1.0
                    self.item.dirty = True
            elif key in (ord("f"), ord("F")):
                self.flagged.append(self.item.stem)
                print(f"  flagged {self.item.stem}")
            elif key in (ord("s"), ord("S")):
                self.item.save()
            elif ord("1") <= key <= ord("9"):
                choice = key - ord("1")
                if choice < len(self.names):
                    if self.selected is not None:
                        self.push_undo()
                        self.item.boxes[self.selected]["cls"] = choice
                        self.item.dirty = True
                    self.current_class = choice
        self.flush()
        cv2.destroyAllWindows()
        print(f"\nreviewed {len(self.reviewed)} images"
              f"{f', flagged {len(self.flagged)}' if self.flagged else ''}")
        print("when finished:  python3 tools/autolabel_dataset.py --split")

    def _mouse_with_offset(self, event, x, y, flags, param):
        self.on_mouse(event, x, y - 66, flags, param)   # status bar height


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--all", action="store_true",
                        help="every image, not just the review queue")
    parser.add_argument("--redo", action="store_true",
                        help="include images already marked reviewed")
    parser.add_argument("--width", type=int, default=1600,
                        help="window width (the image is scaled to fit)")
    args = parser.parse_args()

    root = Path(args.root).expanduser() / "labeled"
    if not (root / "images").is_dir():
        raise SystemExit(f"no labels in {root} - run autolabel_dataset.py first")
    names = [line for line in (root / "classes.txt").read_text().split("\n")
             if line.strip()]

    if args.all:
        entries = [(p.stem, "") for p in sorted((root / "images").glob("*.jpg"))]
    else:
        queue = root / "review_first.txt"
        entries = []
        for line in queue.read_text().split("\n"):
            if not line.strip():
                continue
            parts = line.split(None, 1)
            entries.append((Path(parts[0]).stem,
                            parts[1].strip() if len(parts) > 1 else ""))
    done = set()
    if not args.redo and (root / "reviewed.txt").is_file():
        done = {line.strip() for line
                in (root / "reviewed.txt").read_text().split("\n")
                if line.strip()}
    items = [Item(root, stem, reason) for stem, reason in entries
             if stem not in done]
    if not items:
        print("everything in the queue is already reviewed (--redo to go again)")
        return
    print(f"{len(items)} images to review"
          + (f" ({len(done)} already done)" if done else ""))
    Reviewer(root, items, names, scale_to=args.width).run()


if __name__ == "__main__":
    main()
