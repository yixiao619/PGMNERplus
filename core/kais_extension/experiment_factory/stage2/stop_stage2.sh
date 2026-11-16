#!/usr/bin/env bash
set -euo pipefail
S=/home/jyx/yixiao/MINER-My/kais_extension/experiment_factory/stage2
SUB=$(cat "$S/latest_submission.txt")
printf 'stop scheduling new jobs; active Factory jobs finish normally\n' > "$SUB/stop.request"
echo "Stop requested for $SUB"
