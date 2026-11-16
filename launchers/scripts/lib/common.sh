#!/usr/bin/env bash
# Resolve MINER-My path. Clone it next to this repo or set MINER_MY_ROOT.
set -euo pipefail

PGMNER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PGMNER_ROOT

if [[ -n "${MINER_MY_ROOT:-}" ]]; then
  MINER_MY_ROOT="$(cd "$MINER_MY_ROOT" && pwd)"
else
  MINER_MY_ROOT="$(cd "$PGMNER_ROOT/../MINER-My" && pwd)"
fi
export MINER_MY_ROOT

if [[ ! -f "$MINER_MY_ROOT/main-mner-robert.py" ]]; then
  echo "找不到 MINER-My: $MINER_MY_ROOT" >&2
  echo "export MINER_MY_ROOT=/path/to/MINER-My" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
