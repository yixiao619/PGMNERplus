#!/usr/bin/env bash
# Twitter-2015 主实验（与 run_twitter2017_main.sh 对称）。需先按 data/twitter2015/README.txt 放好数据。
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# export HF_ENDPOINT=https://hf-mirror.com

mkdir -p checkpoints/twitter2015-parmner

python main-mner-robert.py \
  --data_dir data/twitter2015 \
  --output_dir checkpoints/twitter2015-parmner \
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
  --do_predict true
