#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jyx/yixiao/MINER-My
S="$ROOT/kais_extension/experiment_factory/stage3"
if [[ -f "$S/runner.pid" ]] && kill -0 "$(cat "$S/runner.pid")" 2>/dev/null; then echo "Stage3 already running: $(cat "$S/runner.pid")"; exit 1; fi
SUB=$(python "$S/scripts/prepare_stage3.py")
python "$S/scripts/summarize_stage3.py" --submission "$SUB"
setsid -f python -u "$S/scripts/execute_stage3.py" --submission "$SUB" >> "$S/logs/stage3_master.log" 2>&1 < /dev/null
sleep 2
echo "runner_pid=$(cat "$S/runner.pid")"
echo "submission=$SUB"
echo "log=$S/logs/stage3_master.log"
