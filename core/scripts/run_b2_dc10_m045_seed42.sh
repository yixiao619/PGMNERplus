#!/usr/bin/env bash
# C2/D 延伸: dc=1.0, mass=0.45, seed=42
set -euo pipefail
export B2_TAG=dc10_m045 B2_SEED=42 B2_DC=1.0 B2_MASS=0.45
exec bash "$(dirname "$0")/run_pp_ib_b2_cell.sh"
