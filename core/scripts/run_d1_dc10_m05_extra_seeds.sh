#!/usr/bin/env bash
# Extra seeds for dc10_m05 (dust=1.0, mass=0.5). Skips dirs that already finished.
# Usage:
#   bash scripts/run_d1_dc10_m05_extra_seeds.sh
#   SEEDS="789 1024 2024" GPU_ID=0 bash scripts/run_d1_dc10_m05_extra_seeds.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

SEEDS=(789 1024 2024)
if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
  read -r -a SEEDS <<<"$SEEDS_OVERRIDE"
fi
GPU_ID="${GPU_ID:-0}"
LOG="${LOG:-logs/dc10_m05_extra_seeds.log}"
mkdir -p logs

{
  echo "=== dc10_m05 extra seeds start $(date -Is) GPU=$GPU_ID seeds=${SEEDS[*]} ==="
  for seed in "${SEEDS[@]}"; do
    out="outputs/pp_ib_partial_ot_grid_dc10_m05_seed${seed}"
    if [[ -f "$out/eval_results.txt" ]] && grep -q 'Eval results test best ckpt' "$out/eval_results.txt" 2>/dev/null; then
      f1=$(grep -v 'it/s\]' "$out/train.log" 2>/dev/null | grep 'best_score span f1' | tail -1 | awk '{print $NF}' || echo "?")
      echo "[SKIP] seed=$seed already done F1=$f1"
      continue
    fi
    echo ""
    echo "===== dc10_m05 seed=$seed $(date -Is) ====="
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
      B2_TAG=dc10_m05 B2_SEED="$seed" B2_DC=1.0 B2_MASS=0.5 \
      B2_OUT="$out" \
      bash scripts/run_pp_ib_b2_cell.sh
    f1=$(grep -v 'it/s\]' "$out/train.log" 2>/dev/null | grep 'best_score span f1' | tail -1 | awk '{print $NF}' || echo "MISSING")
    echo "seed=$seed test_F1=$f1"
  done
  echo ""
  echo "=== summary $(date -Is) ==="
  python3 scripts/summarize_partial_ot_multiseed.py --tags dc10_m05
  echo "[DONE] dc10_m05 extra seeds $(date -Is)"
} 2>&1 | tee -a "$LOG"
