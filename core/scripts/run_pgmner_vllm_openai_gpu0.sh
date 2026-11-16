#!/usr/bin/env bash
# GPU0: OpenAI VLLM descriptions -> PGMNER+VLLM train (3 seeds)
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
# Default GPU 1 (GPU 0 often busy on shared machines)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

DATASET="${1:-twitter2017}"
LOG="$ROOT/logs/pgmner_vllm_openai_pipeline.log"
GEN_LOG="$ROOT/logs/vllm_desc_${DATASET}_openai.log"

exec > >(tee -a "$LOG") 2>&1
echo "===== pipeline start $(date -Is) dataset=$DATASET GPU=$CUDA_VISIBLE_DEVICES ====="

source "$(dirname "$0")/load_openai_key.sh" 2>/dev/null || true
python3 -c "from openai import OpenAI" 2>/dev/null || {
  echo "ERROR: python3 -m pip install openai"
  exit 1
}
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: 无 OPENAI_API_KEY。请创建 MINER-My/.env.openai（见 .env.openai.example）"
  exit 1
fi

# Stop stale stub generation if any
pkill -f "generate_vllm_desc.py --dataset ${DATASET}.*--backend stub" 2>/dev/null || true

DATA_DIR="$ROOT/data/${DATASET}"
DESC_DIR="$DATA_DIR/vllm_desc"
mkdir -p "$DESC_DIR" logs

# Optional: replace stub cache with OpenAI (set FRESH_OPENAI_DESC=1)
if [[ "${FRESH_OPENAI_DESC:-0}" == "1" ]] && [[ -d "$DESC_DIR" ]]; then
  bak="$ROOT/data/${DATASET}/vllm_desc_stub_$(date +%Y%m%d_%H%M%S)"
  echo "Backing up existing vllm_desc -> $bak"
  mv "$DESC_DIR" "$bak"
  mkdir -p "$DESC_DIR"
fi

echo "----- Phase 1: OpenAI description generation -----"
for split in train dev test; do
  echo ">> generate $split $(date -Is)"
  python3 scripts/generate_vllm_desc.py \
    --dataset "$DATASET" \
    --split "$split" \
    --backend openai \
    --model "${OPENAI_VLLM_MODEL:-gpt-4o-mini}" \
    --qps "${VLLM_QPS:-2}" \
    --data_dir "$DATA_DIR" \
    2>&1 | tee -a "$GEN_LOG"
done

for split in train dev test; do
  f="$DESC_DIR/${split}.jsonl"
  if [[ ! -f "$f" ]] || [[ $(wc -l < "$f") -lt 10 ]]; then
    echo "ERROR: generation incomplete for $split ($f)"
    exit 1
  fi
done

echo "----- Phase 2: train PGMNER+VLLM (seeds ${SEEDS:-42,123,456}) -----"
GPU_ID=0 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  VLLM_BACKEND=openai \
  USE_VLLM_DESC=1 \
  SKIP_GEN=1 \
  SEEDS="${SEEDS:-42,123,456}" \
  bash scripts/run_pgmner_vllm.sh "$DATASET"

echo "===== pipeline done $(date -Is) ====="
python3 scripts/print_pgmner_vllm_latex.py --runs_root runs/pgmner_vllm --datasets "$DATASET"
