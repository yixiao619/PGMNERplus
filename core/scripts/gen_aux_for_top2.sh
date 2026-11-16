#!/usr/bin/env bash
# Fast structured aux (no API): train=gold distill, dev/test=clean+heuristics.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
DATASET="${DATASET:-twitter2017}"
LOG="$ROOT/logs/gen_aux_top2.log"
mkdir -p aux_knowledge logs

# Stop slow stub job if still running
pkill -f "generate_aux.py.*backend stub" 2>/dev/null || true

{
  echo "=== gen_aux_top2 $(date -Is) dataset=$DATASET backend=structured ==="
  python3 aux_knowledge/generate_aux.py \
    --dataset "$DATASET" --split train \
    --aux_mode train_gold_distill --backend structured --qps 50
  for SPLIT in dev test; do
    python3 aux_knowledge/generate_aux.py \
      --dataset "$DATASET" --split "$SPLIT" \
      --aux_mode clean --backend structured --qps 50
  done
  echo "=== counts ==="
  wc -l aux_knowledge/${DATASET}_*_*.jsonl 2>/dev/null || true
  echo "[DONE] $(date -Is)"
} 2>&1 | tee "$LOG"
