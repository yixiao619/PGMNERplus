#!/usr/bin/env bash
export B2_TAG=dc07_m065 B2_SEED=42 B2_DC=0.7 B2_MASS=0.65
exec bash "$(cd "$(dirname "$0")" && pwd)/run_pp_ib_b2_cell.sh"
