#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jyx/yixiao/MINER-My
EXP="$ROOT/kais_extension/experiment24_literature_guided"
cd "$ROOT"
A_PID=482489
echo "[$(date '+%F %T')] waiting for existing Route A PID $A_PID" >> "$EXP/logs/continuation.log"
while kill -0 "$A_PID" 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] Route A exited; continuing B/C without restarting A" >> "$EXP/logs/continuation.log"
E24_SKIP_IDS=E24_A_TASK_GUIDED_OT python -u "$EXP/scripts/scheduler.py" >> "$EXP/logs/continuation.log" 2>&1
python -u "$EXP/scripts/wait_and_run_posthoc.py" >> "$EXP/logs/continuation.log" 2>&1
python "$EXP/scripts/summarize.py" >> "$EXP/logs/continuation.log" 2>&1
echo "[$(date '+%F %T')] Experiment 24 mechanism workflow finished" >> "$EXP/logs/continuation.log"
