#!/usr/bin/env bash
# Round2 实验2：实验1 + span 实体二分类辅助头
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EXP_NAME="${1:-exp_round2_exp2_span_det}"
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
  --span_detection_weight 0.2 \
  --span_head_dropout 0.1 \
  2>&1 | tee "$OUT/train.log"
