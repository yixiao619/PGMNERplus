#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

tmux kill-session -t t17_extra_g0 2>/dev/null || true
tmux kill-session -t t17_extra_g1 2>/dev/null || true

tmux new-session -d -s t17_extra_g0 \
  "bash scripts/run_t17_extra_seeds.sh 0 scripts/t17_extra_seeds_gpu0.keys 2>&1 | tee logs/t17_extra_g0.log; python3 scripts/summarize_t17_ablation_3seed.py >> logs/t17_extra_g0.log 2>&1"

tmux new-session -d -s t17_extra_g1 \
  "bash scripts/run_t17_extra_seeds.sh 1 scripts/t17_extra_seeds_gpu1.keys 2>&1 | tee logs/t17_extra_g1.log; python3 scripts/summarize_t17_ablation_3seed.py >> logs/t17_extra_g1.log 2>&1"

sleep 25
tmux ls
nvidia-smi
tail -3 outputs/t17_extra_seeds_gpu0.summary 2>/dev/null || true
tail -3 outputs/t17_extra_seeds_gpu1.summary 2>/dev/null || true
