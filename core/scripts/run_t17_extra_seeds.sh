#!/usr/bin/env bash
# Usage: bash scripts/run_t17_extra_seeds.sh <GPU> <KEYS_FILE>
# Runs seeds 456,789 for listed t17 ablation keys (seed123 already done in v2 run).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/baseline_hparams.sh"

GPU="${1:?GPU id}"
KEYS_FILE="${2:?keys file}"
MATRIX="scripts/ablation_matrix.tsv"
DATASET="t17"
DATA_DIR="data/twitter2017"
SEEDS=(456 789)
SUMMARY="outputs/t17_extra_seeds_gpu${GPU}.summary"

eval "$(sed -n '/^build_run_args()/,/^}$/p; /^read_test_f1()/,/^}$/p' scripts/run_ablation.sh)"

lookup_extra() {
  local key="$1"
  local k e
  while IFS=$'\t' read -r k e; do
    k="${k//$'\r'/}"
    e="${e//$'\r'/}"
    [[ "$k" == "$key" ]] && { echo "$e"; return 0; }
  done <"$MATRIX"
  return 1
}

echo "[$(date -Is)] extra-seeds start GPU=$GPU keys=$KEYS_FILE" | tee -a "$SUMMARY"

while IFS= read -r key || [[ -n "${key:-}" ]]; do
  key="${key//$'\r'/}"
  [[ -z "$key" || "$key" =~ ^# ]] && continue
  extra="$(lookup_extra "$key")" || { echo "[$key] missing in matrix" | tee -a "$SUMMARY"; continue; }
  echo "===== [t17] $key (seeds ${SEEDS[*]}) =====" | tee -a "$SUMMARY"
  for seed in "${SEEDS[@]}"; do
    out="outputs/${DATASET}_abl_${key}_seed${seed}_v2"
    mkdir -p "$out"
    build_run_args "$key" "${extra:-}"
    echo "[$(date +%H:%M:%S)] launching $key seed=$seed -> $out" | tee -a "$SUMMARY"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 python3 main-mner-robert.py \
      "${RUN_ARGS[@]}" \
      --data_dir "$DATA_DIR" \
      --seed "$seed" \
      --epoch 30 \
      --patience 5 \
      --output_dir "$out" \
      >>"$out/train.log" 2>&1
    rc=$?
    f1="$(read_test_f1 "$out")"
    echo "[$key seed=$seed] rc=$rc test_F1=$f1" | tee -a "$SUMMARY"
    if [[ "$rc" != "0" ]]; then
      echo "[$key seed=$seed] ERROR rc=$rc" | tee -a "$SUMMARY"
      exit 1
    fi
  done
done <"$KEYS_FILE"

echo "[$(date -Is)] extra-seeds GPU=$GPU ALL DONE" | tee -a "$SUMMARY"
