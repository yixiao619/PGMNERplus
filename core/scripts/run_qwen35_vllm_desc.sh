#!/usr/bin/env bash
# Real VLLM descriptions with local Qwen3.5-27B (no OpenAI key).
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_qwen35_vllm_desc.sh twitter2017 dev
#   CUDA_VISIBLE_DEVICES=2 FRESH=1 bash scripts/run_qwen35_vllm_desc.sh twitter2017 train
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

DATASET="${1:-twitter2017}"
SPLIT="${2:-}"
FRESH="${FRESH:-0}"
GPU="${CUDA_VISIBLE_DEVICES:-2}"
MODEL_PATH="${QWEN35_PATH:-/home/LargeModels/Qwen/Qwen3.5-27B}"

export CUDA_VISIBLE_DEVICES="$GPU"
echo "=== Qwen3.5 VLLM desc dataset=$DATASET split=${SPLIT:-all} GPU=$GPU model=$MODEL_PATH ==="

EXTRA=()
[[ "$FRESH" == "1" ]] && EXTRA+=(--fresh)

if [[ -n "$SPLIT" ]]; then
  SPLITS=("$SPLIT")
else
  SPLITS=(train dev test)
fi

for sp in "${SPLITS[@]}"; do
  python3 scripts/generate_vllm_desc.py \
    --dataset "$DATASET" \
    --split "$sp" \
    --backend qwen35 \
    --model_path "$MODEL_PATH" \
    --qps 0 \
    "${EXTRA[@]}" \
    2>&1 | tee -a "logs/qwen35_desc_${DATASET}_${sp}.log"
done

echo "=== done; sample head ==="
head -1 "data/${DATASET}/vllm_desc/dev.jsonl" 2>/dev/null || true
