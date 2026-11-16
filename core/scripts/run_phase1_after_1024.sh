#!/usr/bin/env bash
# 等 seed1024_v2 结束 → A1 汇总 → A2 → A3 → A4（串行，单 GPU）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/logs/phase1_serial.log"
mkdir -p "$ROOT/logs"

exec >>"$LOG" 2>&1
echo ""
echo "=== Phase1 serial start $(date -Is) ==="

while pgrep -f 'main-mner-robert.py.*pp_ib_seed1024_v2' >/dev/null 2>&1; do
  ep=$(grep 'Training epoch num' logs/phase2_pp_ib_v2.log 2>/dev/null | tail -1 || true)
  echo "[wait 1024] $(date -Is) $ep"
  sleep 120
done

if [[ ! -f outputs/pp_ib_seed1024_v2/train.log ]] && [[ -f logs/phase2_pp_ib_v2.log ]]; then
  awk '/PP-IB v2 seed=1024 only/{found=1} found' logs/phase2_pp_ib_v2.log \
    > outputs/pp_ib_seed1024_v2/train.log || true
fi

echo "=== Task A1 summarize $(date -Is) ==="
python3 scripts/summarize_5seed.py

echo "=== Task A2 $(date -Is) ==="
bash scripts/run_pp_ib_partial_ot_seed42.sh

echo "=== Task A3 $(date -Is) ==="
bash scripts/run_pp_ib_partial_ot_ema_seed42.sh

echo "=== Task A4 $(date -Is) ==="
bash scripts/run_pp_ib_partial_ot_ema_focal_seed42.sh

echo "=== Phase1 serial done $(date -Is) ==="
