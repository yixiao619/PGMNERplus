#!/usr/bin/env bash
# D3: dc10_m05 + ema_decay 0.9995 — 仅 D1 判定后人工启动
set -euo pipefail
export B2_TAG=dc10_m05_ema9995 B2_SEED=42 B2_DC=1.0 B2_MASS=0.5
export B2_EMA_DECAY=0.9995
export B2_OUT=outputs/pp_ib_partial_ot_grid_dc10_m05_ema9995_seed42
exec bash "$(dirname "$0")/run_pp_ib_b2_cell.sh"
