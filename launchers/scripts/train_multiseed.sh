#!/usr/bin/env bash
# 多 seed 主实验
# 用法: GPU=0 bash scripts/train_multiseed.sh twitter2017 42 123 456
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DATASET="${1:-twitter2017}"
shift || true
SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=(42 123 456)
fi

for s in "${SEEDS[@]}"; do
  echo "======== seed $s ========"
  GPU="${GPU:-0}" SEED="$s" bash "$ROOT/scripts/train_main.sh" "$DATASET"
done
