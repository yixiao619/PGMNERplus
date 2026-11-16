#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jyx/yixiao/MINER-My
S="$ROOT/kais_extension/experiment_factory/stage2"
if [[ -f "$S/runner.pid" ]] && kill -0 "$(cat "$S/runner.pid")" 2>/dev/null; then echo "Stage2 already running: $(cat "$S/runner.pid")"; exit 1; fi
SUB=$(python "$S/prepare_stage2.py")
mkdir -p "$S/logs"
python "$S/summarize_stage2.py" --submission "$SUB" >/dev/null
setsid -f python -u "$S/execute_stage2.py" --submission "$SUB" >> "$S/logs/stage2_master.log" 2>&1 < /dev/null
sleep 2
PID=$(cat "$S/runner.pid")
echo "runner_pid=$PID"
echo "submission=$SUB"
echo "log=$S/logs/stage2_master.log"
