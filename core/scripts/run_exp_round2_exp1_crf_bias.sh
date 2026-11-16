#!/usr/bin/env bash
# Round2 实验1：仅 span 分类器输出偏置（对应原 CRF 初始化思想），其余同 exp2_pp_ib
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EXP_NAME="${1:-exp_round2_exp1_crf_bias}"
OUT="outputs/${EXP_NAME}"
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
