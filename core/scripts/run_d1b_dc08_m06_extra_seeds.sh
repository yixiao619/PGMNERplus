#!/usr/bin/env bash
# Extra seeds for dc08_m06 (dust=0.8, mass=0.6). Run after dc10_m05 sweep if needed.
# Usage:
#   bash scripts/run_d1b_dc08_m06_extra_seeds.sh
#   SEEDS="789 1024 2024" GPU_ID=0 bash scripts/run_d1b_dc08_m06_extra_seeds.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

SEEDS=(789 1024 2024)
if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
  read -r -a SEEDS <<<"$SEEDS_OVERRIDE"
fi
GPU_ID="${GPU_ID:-0}"
LOG="${LOG:-logs/dc08_m06_extra_seeds.log}"
mkdir -p logs

{
  echo "=== dc08_m06 extra seeds start $(date -Is) GPU=$GPU_ID seeds=${SEEDS[*]} ==="
  for seed in "${SEEDS[@]}"; do
    out="outputs/pp_ib_partial_ot_grid_dc08_m06_seed${seed}"
    if [[ -f "$out/eval_results.txt" ]] && grep -q 'Eval results test best ckpt' "$out/eval_results.txt" 2>/dev/null; then
      f1=$(grep -v 'it/s\]' "$out/train.log" 2>/dev/null | grep 'best_score span f1' | tail -1 | awk '{print $NF}' || echo "?")
      echo "[SKIP] seed=$seed already done F1=$f1"
      continue
    fi
    echo ""
    echo "===== dc08_m06 seed=$seed $(date -Is) ====="
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
      B2_TAG=dc08_m06 B2_SEED="$seed" B2_DC=0.8 B2_MASS=0.6 \
      B2_OUT="$out" \
      bash scripts/run_pp_ib_b2_cell.sh
    f1=$(grep -v 'it/s\]' "$out/train.log" 2>/dev/null | grep 'best_score span f1' | tail -1 | awk '{print $NF}' || echo "MISSING")
    echo "seed=$seed test_F1=$f1"
  done
  echo ""
  echo "=== summary $(date -Is) ==="
  python3 scripts/summarize_partial_ot_multiseed.py --tags dc08_m06
  echo "[DONE] dc08_m06 extra seeds $(date -Is)"
} 2>&1 | tee -a "$LOG"
