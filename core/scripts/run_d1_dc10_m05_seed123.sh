#!/usr/bin/env bash
# D1: dc10_m05 winning config, seed=123
set -euo pipefail
export B2_TAG=dc10_m05 B2_SEED=123 B2_DC=1.0 B2_MASS=0.5
export B2_OUT=outputs/pp_ib_partial_ot_grid_dc10_m05_seed123
exec bash "$(dirname "$0")/run_pp_ib_b2_cell.sh"
