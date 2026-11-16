#!/usr/bin/env bash
set -u
ROOT=/home/jyx/yixiao/MINER-My
EXP="$ROOT/kais_extension/experiment22_module_analysis"
TRAIN_PID="${1:?training PID required}"
echo "$BASHPID" > "$EXP/results/finalize_watcher.pid"
cd "$ROOT"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
python -u "$EXP/scripts/finalize_exp22.py" > "$EXP/logs/finalize.log" 2>&1
echo $? > "$EXP/results/finalize_exit_code.txt"
