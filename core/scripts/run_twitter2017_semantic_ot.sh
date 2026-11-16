#!/usr/bin/env bash
# Semantic OT 替代 CAC（与 run_twitter2017_roberta_base.sh 对齐，仅多 OT 相关 flag）
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p checkpoints/twitter2017-semantic-ot

python main-mner-robert.py \
  --data_dir data/twitter2017 \
  --output_dir checkpoints/twitter2017-semantic-ot \
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
  --do_predict true \
  --use_semantic_ot \
  --ot_alpha 0.05 \
  --ot_eps 0.1 \
  --ot_sinkhorn_iter 10 \
  --ot_lambda_align 1.0
