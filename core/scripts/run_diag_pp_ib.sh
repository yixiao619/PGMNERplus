#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
GPU="${1:-2}"
OUT="${2:-results/diag_pp_ib}"
mkdir -p logs "$OUT"

tmux kill-session -t diag_pp_ib 2>/dev/null || true
tmux new-session -d -s diag_pp_ib \
  "cd '$ROOT' && python3 scripts/diag_pp_ib_protocol.py --gpu $GPU --out_dir $OUT 2>&1 | tee logs/diag_pp_ib.log; echo '[DONE] diag $(date -Is)' | tee -a logs/diag_pp_ib.log"

sleep 3
tmux ls | grep diag || true
tail -5 logs/diag_pp_ib.log 2>/dev/null || true
tail -3 "$OUT/progress.txt" 2>/dev/null || true
