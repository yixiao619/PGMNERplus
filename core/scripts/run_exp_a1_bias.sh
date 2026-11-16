#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${1:-outputs/exp_a1_bias}"
mkdir -p "$OUT" logs
export PYTHONUNBUFFERED=1

python -u main-mner-robert.py \
  --data_dir data/twitter2017 \
  --model_name_or_path pretrain/roberta-large \
  --tokenizer_name pretrain/roberta-large \
  --vit_name pretrain/clip-vit-base-patch16 \
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
  --crf_misc_bias 1.5 \
  --crf_entity_bias 0.5 \
  --crf_o_penalty -0.3 \
  2>&1 | tee "$OUT/train.log"

{
  echo "# exp_a1_bias — $(date -Is)"
  echo "## eval_results.txt"
  [[ -f "$OUT/eval_results.txt" ]] && cat "$OUT/eval_results.txt"
  echo ""
  grep "best_score span f1" "$OUT/train.log" 2>/dev/null | tail -3 || true
} > "$OUT/results.txt"
echo "== Done: $OUT =="
