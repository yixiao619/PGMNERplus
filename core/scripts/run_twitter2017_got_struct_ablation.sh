#!/usr/bin/env bash
# GOT 结构消融：先 identity（A_t/A_v=I），再 ones（全 1）；与 multi_seed / run_twitter2017_got 超参对齐
# 用法：cd MINER-My && bash scripts/run_twitter2017_got_struct_ablation.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs checkpoints/twitter2017-got-identity checkpoints/twitter2017-got-ones

COMMON=(
  --data_dir data/twitter2017
  --model_name_or_path pretrain/roberta-base
  --tokenizer_name pretrain/roberta-base
  --vit_name pretrain/clip-vit-base-patch16
  --gpu_id 0
  --epoch 30
  --batch_size 8
  --max_seq_len 128
  --max_span_len 4
  --lr 3e-5
  --bert_lr 3e-5
  --patience 0  --seed 42
  --logging_steps 100
  --do_train true
  --do_eval true
  --do_predict true
  --use_got
  --ot_alpha 0.05
  --ot_eps 0.1
  --ot_sinkhorn_iter 10
  --ot_lambda_align 1.0
  --got_lambda_sem 0.5
)

echo "========== GOT struct ablation: identity =========="
python main-mner-robert.py \
  "${COMMON[@]}" \
  --got_struct_mode identity \
  --output_dir checkpoints/twitter2017-got-identity \
  > logs/run_twitter2017_got_identity.log 2>&1
echo "OK identity -> checkpoints/twitter2017-got-identity"

echo "========== GOT struct ablation: ones =========="
python main-mner-robert.py \
  "${COMMON[@]}" \
  --got_struct_mode ones \
  --output_dir checkpoints/twitter2017-got-ones \
  > logs/run_twitter2017_got_ones.log 2>&1
echo "OK ones -> checkpoints/twitter2017-got-ones"

echo "All struct ablation runs finished."
