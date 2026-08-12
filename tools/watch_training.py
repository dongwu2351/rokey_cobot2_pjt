#!/usr/bin/env python3
"""Live view of a YOLO training run, readable at a glance.

`tail -f` on the training log shows a redrawing progress bar and little else;
what you actually want to know is whether the score is still climbing and how
long is left.

    python3 tools/watch_training.py            # follows tools_v5
    python3 tools/watch_training.py tools_v4   # another run
    python3 tools/watch_training.py --once     # print once and exit

Ctrl+C stops watching. It never touches the run.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

RUNS = Path.home() / "tool_dataset" / "runs"
COLUMNS = [
    ("epoch", "epoch", "{:>6.0f}"),
    ("train/box_loss", "box_loss", "{:>9.4f}"),
    ("train/cls_loss", "cls_loss", "{:>9.4f}"),
    ("metrics/precision(B)", "P", "{:>7.3f}"),
    ("metrics/recall(B)", "R", "{:>7.3f}"),
    ("metrics/mAP50(B)", "mAP50", "{:>8.4f}"),
    ("metrics/mAP50-95(B)", "mAP50-95", "{:>9.4f}"),
]


def training_alive():
    return subprocess.run(["pgrep", "-f", "yolo detect train"],
                          capture_output=True).returncode == 0


def report(run: str, epochs: int, patience: int):
    csv_path = RUNS / run / "results.csv"
    alive = training_alive()
    lines = [f"=== {run} ===",
             "상태: 학습 중" if alive else "상태: 완료 (또는 중단)"]
    if not csv_path.is_file():
        lines.append("아직 첫 epoch 결과가 없습니다...")
        return "\n".join(lines)

    with open(csv_path) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        lines.append("아직 첫 epoch 결과가 없습니다...")
        return "\n".join(lines)

    key = {name.strip(): name for name in rows[0]}
    value = lambda row, name: float(row[key[name]])
    last = rows[-1]
    done = value(last, "epoch")
    elapsed = value(last, "time")
    best = max(rows, key=lambda r: value(r, "metrics/mAP50(B)"))
    best_epoch = value(best, "epoch")
    stale = int(done - best_epoch)
    remaining = (epochs - done) * (elapsed / max(done, 1)) / 60.0

    lines.append(f"진행 {done:.0f}/{epochs}   경과 {elapsed / 60:.1f}분"
                 + (f"   남은 약 {remaining:.1f}분" if alive else ""))
    lines.append(f"최고 mAP50 {value(best, 'metrics/mAP50(B)'):.4f} "
                 f"@ epoch {best_epoch:.0f}   "
                 f"({stale}/{patience} 개선 없음"
                 + (f" - {patience}이면 조기종료)" if alive else ")"))
    lines.append("")
    lines.append(" ".join(f"{title:>{len(fmt.split(':')[1].split('.')[0].strip('>'))}}"
                          for _, title, fmt in COLUMNS))
    for row in rows[-12:]:
        cells = []
        for name, _, fmt in COLUMNS:
            try:
                cells.append(fmt.format(value(row, name)))
            except (KeyError, ValueError):
                cells.append("".rjust(8))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", default="tools_v5")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()

    if args.once:
        print(report(args.run, args.epochs, args.patience))
        return
    try:
        while True:
            text = report(args.run, args.epochs, args.patience)
            sys.stdout.write("\033[2J\033[H" + text + "\n")
            sys.stdout.flush()
            if not training_alive() and "완료" in text.split("\n")[1]:
                print("\n학습이 끝났습니다. Ctrl+C로 나가세요.")
                time.sleep(args.interval * 4)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
