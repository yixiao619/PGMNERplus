#!/usr/bin/env bash
# Task C 第二步：固定 BEST_BETA（默认 0.005，建议换成 Task B 最优），扫 ot_alpha。
# 用法: BEST_BETA=0.005 bash scripts/run_pp_ib_pot_otalpha_grid.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

BEST_BETA="${BEST_BETA:-0.005}"
for OTA in 0.02 0.05 0.1; do
  OTATAG=$(echo "$OTA" | sed 's/\.//g')
  BETA_POLAR="$BEST_BETA" OT_ALPHA="$OTA" PARTIAL_MASS=0.6 \
    bash scripts/run_pp_ib_partial_ot.sh "outputs/pp_ib_pot_otalpha${OTATAG}"
done
echo "===== ot_alpha grid done (BEST_BETA=$BEST_BETA) ====="
