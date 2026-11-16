#!/usr/bin/env bash
export B2_TAG=dc10_m055 B2_SEED=42 B2_DC=1.0 B2_MASS=0.55
exec bash "$(cd "$(dirname "$0")" && pwd)/run_pp_ib_b2_cell.sh"
