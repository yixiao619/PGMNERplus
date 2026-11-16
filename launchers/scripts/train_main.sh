#!/usr/bin/env bash
# 主实验训练：Twitter-2017 / Twitter-2015
# 用法: GPU=0 SEED=42 bash scripts/train_main.sh twitter2017
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/hparams.sh"

DATASET="${1:-twitter2017}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
EPOCH="${EPOCH:-40}"
PATIENCE="${PATIENCE:-0}"
TAG="${TAG:-dc10_m05}"

DATA_DIR="data/${DATASET}"
OUT="${OUT_DIR:-outputs/pp_ib_partial_ot_grid_${TAG}_seed${SEED}}"

cd "$MINER_MY_ROOT"
mkdir -p "$OUT"

if grep -q 'Eval results test best ckpt' "$OUT/eval_results.txt" 2>/dev/null; then
  echo "skip (已有 test eval): $OUT"
  python3 "$PGMNER_ROOT/scripts/parse_metrics.py" "$OUT/eval_results.txt"
  exit 0
fi

echo "train $DATASET seed=$SEED -> $OUT"
CUDA_VISIBLE_DEVICES="$GPU" python main-mner-robert.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT" \
  --gpu_id 0 \
  --epoch "$EPOCH" \
  --patience "$PATIENCE" \
  --seed "$SEED" \
  --do_train true --do_eval true --do_predict true \
  "${PGMNER_BASE_ARGS[@]}" \
  2>&1 | tee "$OUT/train.log"

python3 "$PGMNER_ROOT/scripts/parse_metrics.py" "$OUT/eval_results.txt"
