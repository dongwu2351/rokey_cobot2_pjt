#!/bin/bash
# Live view of a YOLO run: the last epochs and where the score is going.
#   bash tools/watch_training.sh [run_name]
RUN="${1:-tools_v5}"
CSV="$HOME/tool_dataset/runs/$RUN/results.csv"
watch -n 5 -t "
  echo '=== $RUN ==='
  pgrep -f 'yolo detect train' >/dev/null && echo '상태: 학습 중' || echo '상태: 완료'
  python3 - <<'PY'
import csv
from pathlib import Path
p = Path('$CSV')
if not p.is_file():
    print('아직 첫 epoch 결과 없음...'); raise SystemExit
rows = list(csv.DictReader(open(p)))
k = {c.strip(): c for c in rows[0]}
last = rows[-1]; ep = int(float(last[k['epoch']])); t = float(last[k['time']])
best = max(rows, key=lambda r: float(r[k['metrics/mAP50(B)']]))
be = int(float(best[k['epoch']]))
print(f'진행 {ep}/100   경과 {t/60:.1f}분   남은 약 {(100-ep)*(t/ep)/60:.1f}분')
print(f'최고 mAP50 {float(best[k[\"metrics/mAP50(B)\"]]):.4f} @ epoch {be}   '
      f'({ep-be}/25 개선없음 -> 25면 조기종료)')
print()
print(f'{\"epoch\":>6} {\"box_loss\":>9} {\"cls_loss\":>9} {\"P\":>7} {\"R\":>7} {\"mAP50\":>8} {\"mAP50-95\":>9}')
for r in rows[-12:]:
    print(f\"{int(float(r[k['epoch']])):>6} {float(r[k['train/box_loss']]):>9.4f} \"
          f\"{float(r[k['train/cls_loss']]):>9.4f} {float(r[k['metrics/precision(B)']]):>7.3f} \"
          f\"{float(r[k['metrics/recall(B)']]):>7.3f} {float(r[k['metrics/mAP50(B)']]):>8.4f} \"
          f\"{float(r[k['metrics/mAP50-95(B)']]):>9.4f}\")
PY
"
