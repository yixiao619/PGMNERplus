#!/usr/bin/env bash
# Twitter-2017 · RoBERTa-base + PPOT：β ∈ {0.02, 0.05, 0.2}（参数与 run_twitter2017_ppot_roberta_base.sh 一致，仅改 ppot_beta 与 output_dir）
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

run_one() {
  local beta="$1"
  local out="$2"
  mkdir -p "$out"
  python main-mner-robert.py \
    --data_dir data/twitter2017 \
    --output_dir "$out" \
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
    --use_ppot \
    --ppot_beta "$beta"
}

run_one 0.02 checkpoints/twitter2017-ppot-roberta-base-beta002
run_one 0.05 checkpoints/twitter2017-ppot-roberta-base-beta005
run_one 0.2  checkpoints/twitter2017-ppot-roberta-base-beta02

echo "PPOT beta sweep done. Check eval_results.txt in each output_dir."
