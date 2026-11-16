#!/usr/bin/env bash
# Train PGMNER+VLLM only (Qwen descriptions already cached).
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

DATASET="${1:-twitter2017}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"

echo "PHASE5_TRAIN" >"$ROOT/logs/pgmner_vllm_pipeline.state"
echo "- [进行中] **PGMNER+VLLM 训练 (Qwen desc)** — GPU $GPU seeds 42,123,456 ($(date '+%H:%M:%S'))" >>"$ROOT/logs/pgmner_vllm_TASKS.md"

export SKIP_GEN=1
export USE_VLLM_DESC=1
export VLLM_BACKEND=qwen35
export FORCE_RETRAIN="${FORCE_RETRAIN:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"

bash "$ROOT/scripts/run_pgmner_vllm.sh" "$DATASET"

echo "PHASE6_DONE" >"$ROOT/logs/pgmner_vllm_pipeline.state"
echo "- [完成] **PGMNER+VLLM 训练** — $(date '+%H:%M:%S')" >>"$ROOT/logs/pgmner_vllm_TASKS.md"
