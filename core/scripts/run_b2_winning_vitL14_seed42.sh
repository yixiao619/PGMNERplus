#!/usr/bin/env bash
# Task C3: B2 winning + ViT-L/14 (only if C1 max >= 0.901)
set -euo pipefail
export B2_TAG=dc10_m06_vitL14 B2_SEED=42 B2_DC=1.0 B2_MASS=0.6
export B2_VIT=pretrain/clip-vit-large-patch14 B2_BS=6
export B2_OUT=outputs/pp_ib_partial_ot_grid_dc10_m06_vitL14_seed42
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ ! -f "$ROOT/pretrain/clip-vit-large-patch14/config.json" ]]; then
  echo "[ERROR] Missing pretrain/clip-vit-large-patch14"
  exit 1
fi
exec bash "$(dirname "$0")/run_pp_ib_b2_cell.sh" || {
  echo "[WARN] retry batch_size=4"
  export B2_BS=4
  exec bash "$(dirname "$0")/run_pp_ib_b2_cell.sh"
}
