#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jyx/yixiao/MINER-My
E="$ROOT/kais_extension/experiment23_prediction_coupling"
if [[ -f "$E/runner.pid" ]] && kill -0 "$(cat "$E/runner.pid")" 2>/dev/null; then echo "Experiment 23 already running: $(cat "$E/runner.pid")"; exit 1; fi
setsid -f python -u "$E/scripts/execute.py" >> "$E/logs/experiment23_master.log" 2>&1 < /dev/null
sleep 2
echo "runner_pid=$(cat "$E/runner.pid")"
echo "log=$E/logs/experiment23_master.log"
