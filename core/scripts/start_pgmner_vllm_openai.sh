#!/usr/bin/env bash
# Start OpenAI pipeline (run this in YOUR shell after: export OPENAI_API_KEY=...)
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 尝试从 .env.openai 读取
source "$(dirname "$0")/load_openai_key.sh" 2>/dev/null || true

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "=============================================="
  echo "未找到 OpenAI API Key。请任选一种方式："
  echo "  1) 打开 https://platform.openai.com/api-keys 登录后 Create key"
  echo "  2) cp .env.openai.example .env.openai"
  echo "  3) 编辑 .env.openai，写入: OPENAI_API_KEY=sk-..."
  echo "  4) 再运行: bash scripts/start_pgmner_vllm_openai.sh"
  echo ""
  echo "当前无 key 时，训练已可用 stub 描述在 GPU1 继续跑（见 monitor）。"
  echo "=============================================="
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export VLLM_QPS="${VLLM_QPS:-2}"
export SEEDS="${SEEDS:-42,123,456}"

nohup env OPENAI_API_KEY="$OPENAI_API_KEY" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  VLLM_QPS="$VLLM_QPS" SEEDS="$SEEDS" \
  bash scripts/run_pgmner_vllm_openai_gpu0.sh twitter2017 \
  >> logs/pgmner_vllm_openai_pipeline.log 2>&1 &

echo "Started pipeline PID=$!  GPU=$CUDA_VISIBLE_DEVICES"
echo "Monitor: watch -n 30 bash scripts/monitor_pgmner_vllm.sh twitter2017"
