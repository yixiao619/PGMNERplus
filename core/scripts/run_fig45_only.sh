#!/usr/bin/env bash
# Restart Fig.4/5 only on GPU1 (does not touch fig3_sweep on GPU0).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs figures viz

GPU_VIZ="${1:-1}"

tmux kill-session -t fig45_viz 2>/dev/null || true

tmux new-session -d -s fig45_viz \
  "cd '$ROOT' && \
   python3 scripts/collect_fig4_viz.py --gpu_id $GPU_VIZ 2>&1 | tee logs/fig4_collect.log && \
   python3 scripts/plot_fig4_viz.py 2>&1 | tee logs/fig4_plot.log && \
   python3 scripts/collect_fig5_cases.py --gpu_id $GPU_VIZ 2>&1 | tee logs/fig5_collect.log && \
   python3 scripts/plot_fig5_case.py 2>&1 | tee logs/fig5_plot.log && \
   echo '[DONE] fig45 $(date -Is)' | tee -a logs/fig5_plot.log"

sleep 8
tmux ls 2>/dev/null || true
echo "--- fig4_collect.log ---"
tail -6 logs/fig4_collect.log 2>/dev/null || true
nvidia-smi --query-compute-apps=gpu_index,pid,process_name,used_memory --format=csv 2>/dev/null || nvidia-smi | tail -12
