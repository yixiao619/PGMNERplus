#!/usr/bin/env bash
# Phase2: 14 configs targeting dev F1 > 91.32. Checkpoints on /data (root disk ~99% full).
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p /data/jyx/MINER-My/runs/pgmner_vllm logs/vllm_sweep

export RUNS_BASE="/data/jyx/MINER-My/runs/pgmner_vllm"
export QUEUE="$ROOT/scripts/vllm_experiment_queue_phase2.tsv"
export PHASE=_phase2

echo "" >>logs/vllm_sweep/TASKS.md
echo "## Phase2 ($(date -Is)) target dev F1 > 91.32%" >>logs/vllm_sweep/TASKS.md
echo "- RUNS_BASE=$RUNS_BASE" >>logs/vllm_sweep/TASKS.md

chmod +x scripts/run_vllm_experiment_queue.sh

start_gpu() {
  local g="$1"
  pkill -f "run_vllm_experiment_queue.sh ${g}" 2>/dev/null || true
  sleep 1
  rm -f "logs/vllm_sweep/gpu${g}.lock"
  PHASE=_phase2 RUNS_BASE="$RUNS_BASE" QUEUE="$QUEUE" \
    nohup bash scripts/run_vllm_experiment_queue.sh "$g" >>"logs/vllm_sweep/nohup_gpu${g}_phase2.log" 2>&1 &
  echo "GPU${g} phase2 queue pid=$!"
}

start_gpu 0
start_gpu 2

echo ""
echo "Phase2: 14 experiment groups (see scripts/vllm_experiment_queue_phase2.tsv)"
echo "Monitor: tail -f logs/vllm_sweep/queue_gpu0_phase2.log"
echo "         cat logs/vllm_sweep/BEST.md"
