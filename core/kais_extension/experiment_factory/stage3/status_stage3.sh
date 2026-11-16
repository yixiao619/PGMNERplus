#!/usr/bin/env bash
S=/home/jyx/yixiao/MINER-My/kais_extension/experiment_factory/stage3
cat "$S/runner_status.json" 2>/dev/null || true
SUB=$(cat "$S/latest_submission.txt" 2>/dev/null || true)
[[ -n "$SUB" ]] && tail -40 "$SUB/logs/scheduler.log" 2>/dev/null || true
