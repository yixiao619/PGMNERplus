#!/usr/bin/env bash
# Task B4: B3-style training + CLIP ViT-L/14, seed=42
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="outputs/pp_ib_vitL14_seed42"
VIT_DIR="$ROOT/pretrain/clip-vit-large-patch14"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1

if [[ ! -f "$VIT_DIR/config.json" ]]; then
  echo "[ERROR] Missing $VIT_DIR — run: hf download openai/clip-vit-large-patch14 --local-dir pretrain/clip-vit-large-patch14"
  exit 1
fi

BS=6
python main-mner-robert.py \
  --data_dir data/twitter2017 \
  --model_name_or_path pretrain/roberta-large \
  --tokenizer_name pretrain/roberta-large \
  --vit_name pretrain/clip-vit-large-patch14 \
  --use_hf_clip_vision true --auto_img_dims true \
  --gpu_id 0 --epoch 40 --batch_size "$BS" \
  --max_seq_len 128 --max_span_len 4 \
  --lr 3e-5 --bert_lr 1e-5 \
  --warmup_ratio 0.1 \
  --patience 0 --seed 42 \
  --logging_steps 100 \
  --do_train true --do_eval true --do_predict true \
  --use_pp_ib --beta_polar 0.01 \
  --delta_minus 0.2 --delta_plus 0.8 \
  --use_ema --ema_decay 0.999 \
  --output_dir "$OUT" \
  2>&1 | tee "$OUT/train.log" || {
    if [[ "$BS" -gt 4 ]]; then
      echo "[WARN] OOM or failure at batch_size=$BS — retry with batch_size=4"
      BS=4
      python main-mner-robert.py \
        --data_dir data/twitter2017 \
        --model_name_or_path pretrain/roberta-large \
        --tokenizer_name pretrain/roberta-large \
        --vit_name pretrain/clip-vit-large-patch14 \
        --use_hf_clip_vision true --auto_img_dims true \
        --gpu_id 0 --epoch 40 --batch_size "$BS" \
        --max_seq_len 128 --max_span_len 4 \
        --lr 3e-5 --bert_lr 1e-5 \
        --warmup_ratio 0.1 \
        --patience 0 --seed 42 \
        --logging_steps 100 \
        --do_train true --do_eval true --do_predict true \
        --use_pp_ib --beta_polar 0.01 \
        --delta_minus 0.2 --delta_plus 0.8 \
        --use_ema --ema_decay 0.999 \
        --output_dir "$OUT" \
        2>&1 | tee -a "$OUT/train.log"
    else
      exit 1
    fi
  }

echo ""
grep -v "it/s\]" "$OUT/train.log" | grep "best_score span f1" | tail -3 || true
