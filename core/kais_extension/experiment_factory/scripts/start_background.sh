#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jyx/yixiao/MINER-My
FACTORY="$ROOT/kais_extension/experiment_factory"
SUBMISSION=${1:-$(cat "$FACTORY/runs/latest_submission.txt")}
mkdir -p "$SUBMISSION/logs"
setsid -f python -u "$FACTORY/scheduler.py" --submission "$SUBMISSION" >> "$SUBMISSION/logs/nohup_scheduler.log" 2>&1 < /dev/null
sleep 1
cat "$SUBMISSION/scheduler.pid"
