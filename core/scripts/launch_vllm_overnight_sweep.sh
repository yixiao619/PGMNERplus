#!/usr/bin/env bash
# VLLM-only F1 sweep. GPU1 keeps the active twitter2017/ main job; 0/2 run the queue.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/vllm_sweep

echo "# VLLM F1 sweep ($(date -Is))" >logs/vllm_sweep/TASKS.md
echo "- 只跑 **VLLM+描述**；不跑 novllm" >>logs/vllm_sweep/TASKS.md
echo "- **GPU1**: 现有 \`runs/pgmner_vllm/twitter2017/\` (dust=1.0, seeds 42→456)" >>logs/vllm_sweep/TASKS.md

if pgrep -f 'twitter2017_novllm' >/dev/null; then
  echo "Stopping novllm (not needed)..."
  pkill -f 'twitter2017_novllm' || true
  sleep 2
fi

chmod +x scripts/run_vllm_experiment_queue.sh scripts/collect_vllm_sweep.py

start_gpu() {
  local g="$1"
  if pgrep -f "run_vllm_experiment_queue.sh ${g}" >/dev/null; then
    echo "Restart GPU${g} queue worker"
    pkill -f "run_vllm_experiment_queue.sh ${g}" || true
    sleep 2
  fi
  nohup bash scripts/run_vllm_experiment_queue.sh "$g" >>"logs/vllm_sweep/nohup_gpu${g}.log" 2>&1 &
  echo "Started GPU${g} VLLM queue pid=$!"
}

# Do not start GPU1 queue — avoids OOM with the main training job on GPU1
start_gpu 0

if pgrep -f 'run_vllm_experiment_queue.sh 2' >/dev/null; then
  echo "GPU2 queue already running (leave it; will pick new rows from updated tsv)"
else
  start_gpu 2
fi

echo ""
echo "Best F1:  cat logs/vllm_sweep/BEST.md"
echo "Summary:  cat logs/vllm_sweep/summary_$(date +%Y%m%d).md"
