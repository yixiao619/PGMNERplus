#!/usr/bin/env bash
export B2_TAG=dc08_m06 B2_SEED=42 B2_DC=0.8 B2_MASS=0.6
exec bash "$(cd "$(dirname "$0")" && pwd)/run_pp_ib_b2_cell.sh"
