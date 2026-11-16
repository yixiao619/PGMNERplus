#!/usr/bin/env bash
# Run Qwen3.5 VLLM descriptions for dev, test, train sequentially on one GPU.
# Usage:
#   nohup env CUDA_VISIBLE_DEVICES=2 FRESH=1 bash scripts/run_qwen35_vllm_desc_all.sh \
#     > logs/qwen35_desc_all_nohup.log 2>&1 &
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

DATASET="${1:-twitter2017}"
GPU="${CUDA_VISIBLE_DEVICES:-2}"
export CUDA_VISIBLE_DEVICES="$GPU"
FRESH="${FRESH:-1}"
export FRESH

RUN="$ROOT/scripts/run_qwen35_vllm_desc.sh"

for SPLIT in dev test train; do
  echo "=== queue: starting $DATASET $SPLIT at $(date -Is) ==="
  if [[ "$SPLIT" == "dev" ]]; then
    FRESH=0 bash "$RUN" "$DATASET" "$SPLIT"
  else
    FRESH=1 bash "$RUN" "$DATASET" "$SPLIT"
  fi
  echo "=== queue: finished $DATASET $SPLIT at $(date -Is) ==="
done

echo "=== all splits done for $DATASET ==="
