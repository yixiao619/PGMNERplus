#!/usr/bin/env bash
# D1b: dc08_m06 config, seed=456
set -euo pipefail
export B2_TAG=dc08_m06 B2_SEED=456 B2_DC=0.8 B2_MASS=0.6
export B2_OUT=outputs/pp_ib_partial_ot_grid_dc08_m06_seed456
exec bash "$(dirname "$0")/run_pp_ib_b2_cell.sh"
