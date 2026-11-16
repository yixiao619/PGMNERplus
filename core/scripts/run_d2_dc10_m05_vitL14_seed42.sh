#!/usr/bin/env bash
# D2: dc10_m05 + ViT-L/14 — 仅 D1 判定后人工启动
set -euo pipefail
export B2_TAG=dc10_m05_vitL14 B2_SEED=42 B2_DC=1.0 B2_MASS=0.5
export B2_VIT=pretrain/clip-vit-large-patch14 B2_BS=6
export B2_OUT=outputs/pp_ib_partial_ot_grid_dc10_m05_vitL14_seed42
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ -f "$ROOT/pretrain/clip-vit-large-patch14/config.json" ]] || {
  echo "[ERROR] download clip-vit-large-patch14 first"; exit 1; }
exec bash "$(dirname "$0")/run_pp_ib_b2_cell.sh" || {
  export B2_BS=4
  exec bash "$(dirname "$0")/run_pp_ib_b2_cell.sh"
}
