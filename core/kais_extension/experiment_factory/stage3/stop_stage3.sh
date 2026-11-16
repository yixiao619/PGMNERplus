#!/usr/bin/env bash
S=/home/jyx/yixiao/MINER-My/kais_extension/experiment_factory/stage3
SUB=$(cat "$S/latest_submission.txt")
touch "$SUB/stop.request"
echo "Stop requested. No new jobs will launch; active jobs are allowed to finish."
