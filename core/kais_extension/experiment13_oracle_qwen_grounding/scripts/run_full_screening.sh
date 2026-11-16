#!/usr/bin/env bash
set -euo pipefail

cd /home/jyx

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="/data/cyc/LLARA/peft/src:${PYTHONPATH:-}"

python yixiao/MINER-My/kais_extension/experiment13_oracle_qwen_grounding/scripts/run_oracle_qwen_grounding.py \
  --mode full \
  --device cuda:0 \
  --bf16
