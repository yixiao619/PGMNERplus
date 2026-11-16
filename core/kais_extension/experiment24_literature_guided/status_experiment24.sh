#!/usr/bin/env bash
set -euo pipefail
cd /home/jyx/yixiao/MINER-My
python kais_extension/experiment24_literature_guided/scripts/summarize.py
tail -n 60 kais_extension/experiment24_literature_guided/logs/scheduler.log
