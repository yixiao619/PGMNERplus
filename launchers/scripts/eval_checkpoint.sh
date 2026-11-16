#!/usr/bin/env bash
# 已有 checkpoint 上跑 test（或 dev+test）
# 用法: GPU=0 bash scripts/eval_checkpoint.sh outputs/pp_ib_partial_ot_grid_dc10_m05_seed42
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/hparams.sh"

OUT="${1:?output dir}"
SPLIT="${SPLIT:-both}"   # dev | test | both
GPU="${GPU:-0}"

[[ -f "$OUT/best_model.pth" ]] || { echo "no checkpoint: $OUT/best_model.pth" >&2; exit 1; }

DO_EVAL=false
DO_PRED=false
case "$SPLIT" in
  dev)   DO_EVAL=true ;;
  test)  DO_PRED=true ;;
  both)  DO_EVAL=true; DO_PRED=true ;;
  *) echo "SPLIT=dev|test|both" >&2; exit 1 ;;
esac

cd "$MINER_MY_ROOT"
DATA_DIR="${DATA_DIR:-data/twitter2017}"
[[ "$OUT" == *twitter2015* ]] && DATA_DIR=data/twitter2015

CUDA_VISIBLE_DEVICES="$GPU" python main-mner-robert.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT" \
  --gpu_id 0 \
  --seed 42 \
  --do_train false \
  --do_eval "$DO_EVAL" \
  --do_predict "$DO_PRED" \
  "${PGMNER_BASE_ARGS[@]}" \
  2>&1 | tee -a "$OUT/eval_only.log"

python3 "$PGMNER_ROOT/scripts/parse_metrics.py" "$OUT/eval_results.txt" --per-class
