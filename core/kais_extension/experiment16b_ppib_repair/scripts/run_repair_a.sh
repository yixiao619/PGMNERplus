#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/jyx/yixiao/MINER-My"
EXP="$ROOT/kais_extension/experiment16b_ppib_repair"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-2}"
export PYTHONUNBUFFERED=1
python kais_extension/experiment16b_ppib_repair/scripts/run_repair.py --variant repair_a \
  2>&1 | tee "$EXP/logs/repair_a_train.log"
