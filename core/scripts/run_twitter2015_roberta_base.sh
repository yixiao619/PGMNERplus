#!/usr/bin/env bash
# Twitter-2015 · RoBERTa-base + CLIP（前提：data/twitter2015_images 下 IMGID.jpg 覆盖率≈100%）
# 跑前执行：python scripts/check_image_coverage.py --data_dir data/twitter2015
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p checkpoints/twitter2015-parmner-base

python main-mner-robert.py \
  --data_dir data/twitter2015 \
  --output_dir checkpoints/twitter2015-parmner-base \
  --model_name_or_path pretrain/roberta-base \
  --tokenizer_name pretrain/roberta-base \
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
  --do_predict true
