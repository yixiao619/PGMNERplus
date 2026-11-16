#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/jyx/yixiao/MINER-My"
cd "$ROOT"
GPU_ID="${1:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
LOG="kais_extension/experiment6_checkpoint_vee/logs/experiment6_sequence_gpu${GPU_ID}.log"
echo "===== Experiment 6 launch $(date -Is) physical_gpu=${GPU_ID} =====" | tee "$LOG"
nvidia-smi | tee -a "$LOG"
python - <<'PY' | tee -a "$LOG"
import torch
print('cuda available', torch.cuda.is_available())
print('device count', torch.cuda.device_count())
if torch.cuda.is_available():
    print('logical cuda:0', torch.cuda.get_device_name(0))
PY
python kais_extension/experiment6_checkpoint_vee/scripts/run_experiment6_checkpoint_vee.py \
  2>&1 | tee -a "$LOG"
echo "===== Experiment 6 done $(date -Is) =====" | tee -a "$LOG"
