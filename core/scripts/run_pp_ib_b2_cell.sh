#!/usr/bin/env bash
# B2 winning-style run: PP-IB + partial OT + EMA + lr tune (env-driven)
# Required: B2_TAG B2_SEED B2_DC B2_MASS
# Optional: B2_VIT B2_BS B2_EPOCH B2_EMA_DECAY B2_OUT (override output dir)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

: "${B2_TAG:?set B2_TAG e.g. dc10_m06}"
: "${B2_SEED:?set B2_SEED}"
: "${B2_DC:?set B2_DC}"
: "${B2_MASS:?set B2_MASS}"

B2_VIT="${B2_VIT:-pretrain/clip-vit-base-patch16}"
B2_BS="${B2_BS:-16}"
B2_EPOCH="${B2_EPOCH:-40}"
B2_EMA_DECAY="${B2_EMA_DECAY:-0.999}"
OUT="${B2_OUT:-outputs/pp_ib_partial_ot_grid_${B2_TAG}_seed${B2_SEED}}"

if grep -q 'Eval results test best ckpt' "$OUT/eval_results.txt" 2>/dev/null; then
  echo "[SKIP] $OUT already has final eval"
  grep -v 'it/s\]' "$OUT/train.log" 2>/dev/null | grep 'best_score span f1' | tail -1 || true
  exit 0
fi

mkdir -p "$OUT"
echo "===== B2 cell tag=$B2_TAG seed=$B2_SEED dc=$B2_DC mass=$B2_MASS ema=$B2_EMA_DECAY -> $OUT ====="

python main-mner-robert.py \
  --data_dir data/twitter2017 \
  --model_name_or_path pretrain/roberta-large \
  --tokenizer_name pretrain/roberta-large \
  --vit_name "$B2_VIT" \
  --use_hf_clip_vision true --auto_img_dims true \
  --gpu_id "${B2_GPU_ID:-0}" --epoch "$B2_EPOCH" --batch_size "$B2_BS" \
  --max_seq_len 128 --max_span_len 4 \
  --lr 3e-5 --bert_lr 1e-5 \
  --warmup_ratio 0.1 \
  --patience 0 --seed "$B2_SEED" \
  --logging_steps 100 \
  --do_train true --do_eval true --do_predict true \
  --use_pp_ib --beta_polar 0.01 \
  --delta_minus 0.2 --delta_plus 0.8 \
  --use_got --got_mode partial_sinkhorn \
  --ot_alpha 0.05 --entropy_eps 0.1 \
  --ot_sinkhorn_iter 30 \
  --partial_mass_init "$B2_MASS" --dust_cost "$B2_DC" \
  --ot_warmup_epochs 3 \
  --use_ema --ema_decay "$B2_EMA_DECAY" \
  --output_dir "$OUT" \
  2>&1 | tee "$OUT/train.log"

echo ""
grep -v 'it/s\]' "$OUT/train.log" | grep 'best_score span f1' | tail -3 || true
