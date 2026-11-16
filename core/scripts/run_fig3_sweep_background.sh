#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU="${1:-2}"
tmux kill-session -t fig3_sweep_v2 2>/dev/null || true
tmux new-session -d -s fig3_sweep_v2 \
  "cd '$ROOT' && bash scripts/run_fig3_sweep.sh $GPU 2>&1 | tee logs/fig3_sweep_v2.log; \
   echo '[DONE] fig3_sweep_v2 $(date -Is)' | tee -a logs/fig3_sweep_v2.log"
sleep 3
tmux ls | grep fig3_sweep_v2 || true
tail -5 "$ROOT/logs/fig3_sweep_v2.log" 2>/dev/null || true
