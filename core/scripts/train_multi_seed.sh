#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs/ensemble

GPU_ID="${GPU_ID:-0}"
VIT_NAME="${VIT_NAME:-pretrain/clip-vit-base-patch16}"
BEST_BETA="${BEST_BETA:-0.01}"
BEST_LR="${BEST_LR:-3e-5}"
# openai id 需联网；本地目录则用 pretrain/clip-vit-large-patch14
USE_HF="${USE_HF_CLIP:-true}"

if [[ ! -d "$VIT_NAME" ]] && [[ "$VIT_NAME" != *"/"* ]]; then
  echo "[WARN] ViT path may be invalid: $VIT_NAME"
fi

EXTRA_CLIP=()
if [[ "$USE_HF" == "true" ]] || [[ "$USE_HF" == "1" ]]; then
  EXTRA_CLIP=(--use_hf_clip_vision true --auto_img_dims true)
fi

SEEDS=(42 123 456 789 1024)

for seed in "${SEEDS[@]}"; do
  echo "=== Training seed=$seed ==="
  OD="outputs/ensemble/seed_${seed}"
  mkdir -p "$OD"
  python main-mner-robert.py \
    --data_dir data/twitter2017 \
    --model_name_or_path pretrain/roberta-large \
    --tokenizer_name pretrain/roberta-large \
    --vit_name "$VIT_NAME" \
    "${EXTRA_CLIP[@]}" \
    --gpu_id "$GPU_ID" \
    --epoch 30 \
    --batch_size 8 \
    --max_seq_len 128 \
    --max_span_len 4 \
    --lr "$BEST_LR" \
    --bert_lr "$BEST_LR" \
    --patience 0 \
    --seed "$seed" \
    --logging_steps 100 \
    --use_pp_ib \
    --beta_polar "$BEST_BETA" \
    --delta_minus 0.2 \
    --delta_plus 0.8 \
    --output_dir "$OD" \
    --do_train true --do_eval true --do_predict true \
    2>&1 | tee "$OD/train.log"
done

echo "All seeds done."
