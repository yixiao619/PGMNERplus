#!/usr/bin/env bash
# Phase 2 D 系列: D1(dc10_m05) + D1b(dc08_m06) 多 seed + D-extra(m045)
# 单独跑: bash scripts/run_phase2_d_series.sh
# C2 后自动链: 由 run_phase2_serial.sh 设置 PHASE2_CHAINED=1 调用
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/logs/phase2_d_series.log"
mkdir -p "$ROOT/logs"
exec >>"$LOG" 2>&1

echo ""
echo "=== Phase 2 D-series start $(date -Is) chained=${PHASE2_CHAINED:-0} ==="

if [[ "${PHASE2_CHAINED:-0}" != "1" ]]; then
  while pgrep -f 'scripts/run_phase2_serial.sh' >/dev/null 2>&1; do
    echo "[wait phase2_serial] $(date -Is)"
    sleep 60
  done
  while pgrep -f 'main-mner-robert.py' >/dev/null 2>&1; do
    echo "[wait GPU] $(date -Is)"
    sleep 60
  done
else
  echo "[chained] skip wait for phase2_serial (called from C2 tail)"
  while pgrep -f 'main-mner-robert.py' >/dev/null 2>&1; do
    echo "[wait GPU] $(date -Is)"
    sleep 30
  done
fi

echo "===== D1 dc10_m05 seed=123 $(date -Is) ====="
bash scripts/run_d1_dc10_m05_seed123.sh

echo "===== D1 dc10_m05 seed=456 $(date -Is) ====="
bash scripts/run_d1_dc10_m05_seed456.sh

echo "===== D1b dc08_m06 seed=123 $(date -Is) ====="
bash scripts/run_d1b_dc08_m06_seed123.sh

echo "===== D1b dc08_m06 seed=456 $(date -Is) ====="
bash scripts/run_d1b_dc08_m06_seed456.sh

echo ""
echo "===== D 系列三 seed 汇总 ====="
echo "--- dc10_m05 ---"
for S in 42 123 456; do
  D="$ROOT/outputs/pp_ib_partial_ot_grid_dc10_m05_seed${S}"
  F1=$(grep -v 'it/s\]' "$D/train.log" 2>/dev/null | grep 'best_score span f1' | tail -1 | awk '{print $NF}' || echo "MISSING")
  echo "seed=${S}: ${F1}"
done
echo "--- dc08_m06 ---"
for S in 42 123 456; do
  D="$ROOT/outputs/pp_ib_partial_ot_grid_dc08_m06_seed${S}"
  F1=$(grep -v 'it/s\]' "$D/train.log" 2>/dev/null | grep 'best_score span f1' | tail -1 | awk '{print $NF}' || echo "MISSING")
  echo "seed=${S}: ${F1}"
done

echo ""
echo "===== D-extra: dc10_m045 (dc=1.0, mass=0.45, seed=42) $(date -Is) ====="
bash scripts/run_b2_dc10_m045_seed42.sh
F1_M045=$(grep -v 'it/s\]' "$ROOT/outputs/pp_ib_partial_ot_grid_dc10_m045_seed42/train.log" 2>/dev/null | grep 'best_score span f1' | tail -1 | awk '{print $NF}' || echo "MISSING")
echo "dc10_m045 seed=42: ${F1_M045}"

echo "=== Write phase2_d1_summary $(date -Is) ==="
python3 scripts/write_phase2_d1_summary.py

echo "=== D-series done (D2/D3 NOT auto-run) $(date -Is) ==="
echo "Await manual decision before run_d2_* / run_d3_*"
