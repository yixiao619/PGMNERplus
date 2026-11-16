#!/usr/bin/env bash
E=/home/jyx/yixiao/MINER-My/kais_extension/experiment23_prediction_coupling
cat "$E/run_state.json" 2>/dev/null || true
tail -40 "$E/logs/experiment23_master.log" 2>/dev/null || true
