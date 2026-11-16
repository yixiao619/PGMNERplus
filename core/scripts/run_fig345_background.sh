#!/usr/bin/env bash
# GPU0: Fig.3 sweep resume | GPU1: Fig.4/5 collect+plot
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs results/runs figures viz

GPU_SWEEP="${1:-0}"
GPU_VIZ="${2:-1}"

tmux kill-session -t fig3_sweep 2>/dev/null || true
tmux kill-session -t fig45_viz 2>/dev/null || true

tmux new-session -d -s fig3_sweep \
  "cd '$ROOT' && python3 scripts/sweep_hparams.py --gpu $GPU_SWEEP 2>&1 | tee -a logs/fig3_sweep.log; \
   python3 scripts/plot_fig3_hparam.py 2>&1 | tee -a logs/fig3_sweep.log; \
   echo '[DONE] fig3 sweep+plot $(date -Is)' | tee -a logs/fig3_sweep.log"

tmux new-session -d -s fig45_viz \
  "cd '$ROOT' && \
   python3 scripts/collect_fig4_viz.py --gpu_id $GPU_VIZ 2>&1 | tee logs/fig4_collect.log && \
   python3 scripts/plot_fig4_viz.py 2>&1 | tee logs/fig4_plot.log && \
   python3 scripts/collect_fig5_cases.py --gpu_id $GPU_VIZ 2>&1 | tee logs/fig5_collect.log && \
   python3 scripts/plot_fig5_case.py 2>&1 | tee logs/fig5_plot.log && \
   echo '[DONE] fig45 $(date -Is)' | tee -a logs/fig5_plot.log"

sleep 12
tmux ls
echo "---"
tail -4 logs/fig3_sweep.log 2>/dev/null || true
tail -4 logs/fig4_collect.log 2>/dev/null || true
nvidia-smi --query-compute-apps=gpu_index,pid,process_name,used_memory --format=csv 2>/dev/null || nvidia-smi | tail -12
