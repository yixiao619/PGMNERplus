#!/usr/bin/env bash
# Task A3: PP-IB + Partial OT + EMA, seed=42
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="outputs/pp_ib_partial_ot_ema_seed42"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1

python main-mner-robert.py \
  --data_dir data/twitter2017 \
  --model_name_or_path pretrain/roberta-large \
  --tokenizer_name pretrain/roberta-large \
  --vit_name pretrain/clip-vit-base-patch16 \
  --use_hf_clip_vision true --auto_img_dims true \
  --gpu_id 0 --epoch 30 --batch_size 8 \
  --max_seq_len 128 --max_span_len 4 \
  --lr 3e-5 --bert_lr 3e-5 \
  --patience 0 --seed 42 \
  --logging_steps 100 \
  --do_train true --do_eval true --do_predict true \
  --use_pp_ib --beta_polar 0.01 \
  --delta_minus 0.2 --delta_plus 0.8 \
  --use_got --got_mode partial_sinkhorn \
  --ot_alpha 0.05 --entropy_eps 0.1 \
  --ot_sinkhorn_iter 30 \
  --partial_mass_init 0.6 --dust_cost 0.5 \
  --ot_warmup_epochs 3 \
  --use_ema \
  --ema_decay 0.999 \
  --output_dir "$OUT" \
  2>&1 | tee "$OUT/train.log"

echo ""
echo "===== Best F1 ====="
grep -v "it/s\]" "$OUT/train.log" | grep "best_score span f1" | tail -3 || true
