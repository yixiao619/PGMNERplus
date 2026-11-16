#!/usr/bin/env bash
# Combine best partial-OT knobs: mass0.7 (90.76%) + dc10_m05 dust=1.0 (90.65%)
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SEED="${SEED:-42}"
OUT="outputs/push915_dc10mass07_seed${SEED}"
mkdir -p "$OUT" logs

if grep -q 'Eval results test best ckpt' "$OUT/eval_results.txt" 2>/dev/null; then
  echo "[SKIP] $OUT done"
  grep -A1 'test best ckpt' "$OUT/eval_results.txt" | grep span_f1
  exit 0
fi

echo "===== push915 train seed=$SEED -> $OUT $(date -Is) ====="
python main-mner-robert.py \
  --data_dir data/twitter2017 \
  --model_name_or_path pretrain/roberta-large \
  --tokenizer_name pretrain/roberta-large \
  --vit_name pretrain/clip-vit-base-patch16 \
  --use_hf_clip_vision true --auto_img_dims true \
  --gpu_id 0 --epoch 40 --batch_size 16 \
  --max_seq_len 128 --max_span_len 4 \
  --lr 3e-5 --bert_lr 1e-5 --warmup_ratio 0.1 \
  --patience 0 --seed "$SEED" --logging_steps 100 \
  --do_train true --do_eval true --do_predict true \
  --use_pp_ib --beta_polar 0.01 --delta_minus 0.2 --delta_plus 0.8 \
  --use_got --got_mode partial_sinkhorn \
  --ot_alpha 0.05 --entropy_eps 0.1 --ot_sinkhorn_iter 30 \
  --partial_mass_init 0.7 --dust_cost 1.0 \
  --ot_warmup_epochs 3 --use_ema --ema_decay 0.999 \
  --output_dir "$OUT" \
  2>&1 | tee "$OUT/train.log"

grep -A1 'test best ckpt' "$OUT/eval_results.txt" 2>/dev/null | grep span_f1 || true
