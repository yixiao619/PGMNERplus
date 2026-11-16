#!/usr/bin/env bash
# PP-IB + Partial OT（--use_got + partial_sinkhorn），与 GOT-CAC 分支独立
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

BETA_POLAR="${BETA_POLAR:-0.005}"
OT_ALPHA="${OT_ALPHA:-0.05}"
PARTIAL_MASS="${PARTIAL_MASS:-0.6}"
OUT="${1:-outputs/pp_ib_partial_ot}"

if [[ -f "$OUT/eval_results.txt" ]]; then
  echo "[SKIP] $OUT 已有 eval_results.txt"
  exit 0
fi

mkdir -p "$OUT"
echo "===== PP-IB + Partial OT  β_polar=$BETA_POLAR  ot_alpha=$OT_ALPHA  partial_mass=$PARTIAL_MASS  out=$OUT ====="
python main-mner-robert.py \
  --data_dir data/twitter2017 \
  --model_name_or_path pretrain/roberta-large \
  --tokenizer_name pretrain/roberta-large \
  --vit_name pretrain/clip-vit-base-patch16 \
  --use_hf_clip_vision true \
  --auto_img_dims true \
  --gpu_id 0 --epoch 30 --batch_size 8 \
  --max_seq_len 128 --max_span_len 4 \
  --lr 3e-5 --bert_lr 3e-5 \
  --patience 0 --seed 42 \
  --logging_steps 100 \
  --do_train true --do_eval true --do_predict true \
  --use_pp_ib \
  --beta_polar "$BETA_POLAR" \
  --delta_minus 0.2 --delta_plus 0.8 \
  --use_got \
  --got_mode partial_sinkhorn \
  --ot_alpha "$OT_ALPHA" \
  --entropy_eps 0.1 \
  --ot_sinkhorn_iter 30 \
  --partial_mass_init "$PARTIAL_MASS" \
  --dust_cost 0.5 \
  --ot_warmup_epochs 3 \
  --output_dir "$OUT" \
  2>&1 | tee "$OUT/train.log"

echo ""
echo "===== Best F1 ====="
grep "best_score span f1" "$OUT/train.log" 2>/dev/null | tail -3 || true
