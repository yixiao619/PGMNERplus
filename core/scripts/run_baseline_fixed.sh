#!/usr/bin/env bash
# 修复 vision encoder bug 后的 baseline 复现脚本
# 用法: bash scripts/run_baseline_fixed.sh [输出目录]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${1:-outputs/baseline_fixed_b16}"
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
  2>&1 | tee "$OUT/train.log"

echo ""
echo "===== Best F1 ====="
grep "best_score span f1" "$OUT/train.log" 2>/dev/null | tail -3 || true
