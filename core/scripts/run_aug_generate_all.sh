#!/usr/bin/env bash
# Generate cached aux knowledge for T-2017 (all splits, clean mode). Resumable.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

BACKEND="${BACKEND:-stub}"
DATASET="${DATASET:-twitter2017}"
LOG="$ROOT/logs/aug_generate_${DATASET}.log"
mkdir -p logs aux_knowledge

{
  echo "=== aug generate start $(date -Is) backend=$BACKEND dataset=$DATASET ==="
  for SPLIT in train dev test; do
    echo "--- split=$SPLIT ---"
    python3 aux_knowledge/generate_aux.py \
      --dataset "$DATASET" \
      --split "$SPLIT" \
      --aux_mode clean \
      --backend "$BACKEND" \
      --qps "${QPS:-5}"
  done
  echo "[DONE] $(date -Is)"
} 2>&1 | tee -a "$LOG"
