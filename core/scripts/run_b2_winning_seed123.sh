#!/usr/bin/env bash
set -euo pipefail
export B2_TAG=dc10_m06 B2_SEED=123 B2_DC=1.0 B2_MASS=0.6
exec bash "$(dirname "$0")/run_pp_ib_b2_cell.sh"
