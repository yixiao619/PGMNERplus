#!/usr/bin/env bash
# Fast push to 91.5%: hard-vote top checkpoints + official span_f1 eval
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

OUT="outputs/ensemble/push915_all8"
mkdir -p "$OUT" logs

MODEL_DIRS=(
  "outputs/t17_fig3_mass0.7_dc08_seed42"
  "outputs/pp_ib_partial_ot_grid_dc10_m05_seed42"
  "outputs/pp_ib_partial_ot_grid_dc08_m06_seed42"
  "outputs/pp_ib_partial_ot_grid_dc08_m06_seed123"
  "outputs/t17_abl_noppib_seed456"
  "outputs/exp2_pp_ib"
  "outputs/t17_fig3_mass0.3_dc08_seed42"
  "outputs/dyn_cosine"
)

DIRS_CSV=$(IFS=,; echo "${MODEL_DIRS[*]}")
python scripts/ensemble_predict.py \
  --model_dirs "$DIRS_CSV" \
  --output "$OUT/test_predictions_voted.txt" \
  2>&1 | tee "$OUT/ensemble.log"

python scripts/eval_ensemble_span.py \
  --predictions "$OUT/test_predictions_voted.txt" \
  --ref_output_dir outputs/t17_fig3_mass0.7_dc08_seed42 \
  --gpu_id 0 \
  2>&1 | tee "$OUT/official_span_f1.log"

echo "Done -> $OUT"
