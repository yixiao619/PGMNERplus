#!/usr/bin/env bash
# Twitter-2017 · RoBERTa-base + CLIP + 写死 OT/IB/SCL（与 main-mner-robert.py CLI 一致）
# 无 --dataset；测试集用 --do_predict true（不是 --do_test）
# 需已存在：pretrain/roberta-base、pretrain/clip-vit-base-patch16（或改为 Hub id）
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p checkpoints/twitter2017-parmner-base

python main-mner-robert.py \
  --data_dir data/twitter2017 \
  --output_dir checkpoints/twitter2017-parmner-base \
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
