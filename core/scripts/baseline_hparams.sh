#!/usr/bin/env bash
# 0.9064 baseline (pp_ib_partial_ot_grid_dc10_m05_seed42, test F1=0.90647)
# dust_cost=0.8, partial_mass_init=0.5, PP-IB + partial_sinkhorn + EMA
BASELINE_ARGS=(
  --model_name_or_path pretrain/roberta-large
  --tokenizer_name pretrain/roberta-large
  --vit_name pretrain/clip-vit-base-patch16
  --use_hf_clip_vision true
  --auto_img_dims true
  --gpu_id 0
  --batch_size 16
  --max_seq_len 128
  --max_span_len 4
  --lr 3e-5
  --bert_lr 1e-5
  --warmup_ratio 0.1
  --logging_steps 100
  --do_train true
  --do_eval true
  --do_predict true
  --use_pp_ib
  --beta_polar 0.01
  --delta_minus 0.2
  --delta_plus 0.8
  --use_got
  --got_mode partial_sinkhorn
  --ot_alpha 0.05
  --entropy_eps 0.1
  --ot_sinkhorn_iter 30
  --partial_mass_init 0.5
  --dust_cost 0.8
  --ot_warmup_epochs 3
  --use_ema
  --ema_decay 0.999
)
