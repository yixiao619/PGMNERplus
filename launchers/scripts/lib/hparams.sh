#!/usr/bin/env bash
# dc10_m05 默认超参，train / eval / ablation 共用
PGMNER_BASE_ARGS=(
  --model_name_or_path pretrain/roberta-large
  --tokenizer_name pretrain/roberta-large
  --vit_name pretrain/clip-vit-base-patch16
  --use_hf_clip_vision true
  --auto_img_dims true
  --batch_size 16
  --max_seq_len 128
  --max_span_len 4
  --lr 3e-5
  --bert_lr 1e-5
  --warmup_ratio 0.1
  --logging_steps 100
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
