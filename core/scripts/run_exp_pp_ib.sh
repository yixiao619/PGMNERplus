#!/usr/bin/env bash
# 对照组 2：仅 PP-IB
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${1:-outputs/exp2_pp_ib}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1

python main-mner-robert.py \
  --data_dir data/twitter2017 \
  --model_name_or_path pretrain/roberta-large \
  --tokenizer_name pretrain/roberta-large \
  --vit_name pretrain/clip-vit-base-patch16 \
  --use_hf_clip_vision true \
  --auto_img_dims true \
  --gpu_id 0 \
  --epoch 30 \
  --batch_size 8 \
  --max_seq_len 128 \
  --max_span_len 4 \
  --lr 3e-5 \
  --bert_lr 3e-5 \
  --patience 0 \
  --seed 42 \
  --logging_steps 100 \
  --do_train true \
  --do_eval true \
  --do_predict true \
  --output_dir "$OUT" \
  --use_pp_ib \
  --beta_polar 0.01 \
  --delta_minus 0.2 \
  --delta_plus 0.8 \
  2>&1 | tee "$OUT/train.log"

{
  echo "# results.txt — $(date -Is)"
  echo "## eval_results.txt"
  if [[ -f "$OUT/eval_results.txt" ]]; then
    cat "$OUT/eval_results.txt"
  fi
  echo ""
  grep "best_score span f1" "$OUT/train.log" 2>/dev/null | tail -3 || true
} >"$OUT/results.txt"
