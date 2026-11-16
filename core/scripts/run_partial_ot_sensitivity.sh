#!/usr/bin/env bash
# partial_sinkhorn 的 eps 与 partial_mass_init 网格（缩短 epoch 时请自行改 --epoch）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${OUT:-$ROOT/outputs/partial_ot_sensitivity}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1

COMMON=(
  --data_dir data/twitter2017
  --model_name_or_path pretrain/roberta-large
  --tokenizer_name pretrain/roberta-large
  --vit_name pretrain/clip-vit-base-patch16
  --gpu_id 0
  --epoch 30
  --batch_size 8
  --max_seq_len 128
  --max_span_len 4
  --lr 3e-5
  --bert_lr 3e-5
  --patience 0
  --seed 42
  --logging_steps 100
  --do_train true
  --do_eval true
  --do_predict true
  --use_pp_ib
  --beta_polar 0.01
  --use_got
  --got_ot_backend partial_sinkhorn
  --ot_alpha 0.05
  --ot_sinkhorn_iter 30
  --dust_cost 0.5
)

for EPS in 0.05 0.10 0.20 0.50; do
  for MASS in 0.4 0.5 0.6 0.7 0.8; do
    TAG="eps${EPS}_mass${MASS}"
    python main-mner-robert.py "${COMMON[@]}" \
      --ot_eps "$EPS" \
      --partial_mass_init "$MASS" \
      --output_dir "$OUT/$TAG" \
      2>&1 | tee "$OUT/${TAG}.log"
    echo "[$TAG] $(grep -iE 'span|f1|best' "$OUT/${TAG}.log" | tail -5)"
  done
done
