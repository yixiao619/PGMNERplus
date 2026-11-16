#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jyx/yixiao/MINER-My
EXP="$ROOT/kais_extension/experiment22_module_analysis"
CKPT_ROOT="$EXP/checkpoints"
mkdir -p "$CKPT_ROOT" "$EXP/logs"
cd "$ROOT"

run_one() {
  local id="$1" kl="$2"
  local out="$CKPT_ROOT/$id"
  mkdir -p "$out"
  python -u main-mner-robert.py \
    --data_dir data/twitter2017 --output_dir "$out" \
    --model_name_or_path pretrain/roberta-large --tokenizer_name pretrain/roberta-large \
    --vit_name pretrain/clip-vit-base-patch16 --use_hf_clip_vision true --auto_img_dims true \
    --gpu_id 0 --epoch 5 --batch_size 16 --max_span_len 4 --max_seq_len 128 \
    --lr 3e-5 --bert_lr 1e-5 --weight_decay 0.01 --warmup_ratio 0.1 \
    --patience 0 --seed 42 --logging_steps 100 --eval_split_for_best dev \
    --do_train true --do_eval true --do_predict false --use_ema --ema_decay 0.999 \
    --use_pp_ib --ib_kl_weight "$kl" --beta_polar 0.01 --delta_minus 0.2 --delta_plus 0.8 \
    --pp_ib_lambda_balance 1.0 --beta_polar_warmup_epochs 3 --pp_ib_probe \
    --use_got --got_mode partial_sinkhorn --ot_alpha 0.05 --entropy_eps 0.1 \
    --ot_sinkhorn_iter 30 --partial_mass_init 0.5 --dust_cost 1.0 --ot_warmup_epochs 3 \
    2>&1 | tee "$EXP/logs/${id}.log"
}

run_one full_kl_short 0.001
run_one no_kl_short 0.0
