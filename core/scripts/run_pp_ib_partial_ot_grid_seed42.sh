#!/usr/bin/env bash
# Task B2: partial OT 2x2 micro-grid on B4-style config (run only if B1 >= 0.895)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

VIT="${VIT_NAME:-pretrain/clip-vit-base-patch16}"
BS="${BATCH_SIZE:-16}"
EPOCH="${EPOCH:-40}"

for CONFIG in "1.0 0.8" "1.0 0.6" "0.5 0.8" "2.0 0.7"; do
  DC=$(echo "$CONFIG" | awk '{print $1}')
  MASS=$(echo "$CONFIG" | awk '{print $2}')
  TAG="dc${DC//./}_m${MASS//./}"
  OUT="outputs/pp_ib_partial_ot_grid_${TAG}"
  if grep -q 'Eval results test best ckpt' "$OUT/eval_results.txt" 2>/dev/null; then
    echo "[SKIP] $OUT"
    continue
  fi
  mkdir -p "$OUT"
  echo "===== partial OT grid dust=$DC mass=$MASS -> $OUT ====="
  python main-mner-robert.py \
    --data_dir data/twitter2017 \
    --model_name_or_path pretrain/roberta-large \
    --tokenizer_name pretrain/roberta-large \
    --vit_name "$VIT" \
    --use_hf_clip_vision true --auto_img_dims true \
    --gpu_id 0 --epoch "$EPOCH" --batch_size "$BS" \
    --max_seq_len 128 --max_span_len 4 \
    --lr 3e-5 --bert_lr 1e-5 \
    --warmup_ratio 0.1 \
    --patience 0 --seed 42 \
    --logging_steps 100 \
    --do_train true --do_eval true --do_predict true \
    --use_pp_ib --beta_polar 0.01 \
    --delta_minus 0.2 --delta_plus 0.8 \
    --use_got --got_mode partial_sinkhorn \
    --ot_alpha 0.05 --entropy_eps 0.1 \
    --ot_sinkhorn_iter 30 \
    --partial_mass_init "$MASS" --dust_cost "$DC" \
    --ot_warmup_epochs 3 \
    --use_ema --ema_decay 0.999 \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/train.log"
  grep -v "it/s\]" "$OUT/train.log" | grep "best_score span f1" | tail -1 || true
done

echo "===== B2 grid done ====="
