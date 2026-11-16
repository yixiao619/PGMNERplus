#!/usr/bin/env bash
# Step A: 5-epoch PP-IB probe run (save Fig.4(a) + VERDICT)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
GPU="${1:-2}"
OUT="${ROOT}/results/pp_ib_fix_stepA"
mkdir -p logs "$OUT"

tmux kill-session -t pp_ib_stepA 2>/dev/null || true
tmux new-session -d -s pp_ib_stepA \
  "cd '$ROOT' && python3 main-mner-robert.py \
    --data_dir data/twitter2017 \
    --output_dir '$OUT' \
    --model_name_or_path pretrain/roberta-large \
    --tokenizer_name pretrain/roberta-large \
    --vit_name pretrain/clip-vit-base-patch16 \
    --use_hf_clip_vision true \
    --auto_img_dims true \
    --gpu_id $GPU \
    --batch_size 16 \
    --max_seq_len 128 \
    --max_span_len 4 \
    --lr 3e-5 \
    --bert_lr 1e-5 \
    --warmup_ratio 0.1 \
    --logging_steps 999999 \
    --do_train true \
    --do_eval false \
    --do_predict false \
    --patience 0 \
    --seed 42 \
    --epoch 5 \
    --use_pp_ib \
    --pp_ib_probe \
    --pp_ib_gate_on_hidden \
    --pp_ib_gate_init_std 0.1 \
    --beta_polar 0.01 \
    --kl_weight 0.0002 \
    --delta_minus 0.2 \
    --delta_plus 0.8 \
    --pp_ib_lambda_balance 1.0 \
    --beta_polar_warmup_epochs 0 \
    --use_got \
    --got_mode partial_sinkhorn \
    --ot_alpha 0.2 \
    --entropy_eps 0.1 \
    --partial_mass_init 0.6 \
    --dust_cost 0.5 \
    --ot_sinkhorn_iter 30 \
    --ot_warmup_epochs 3 \
    2>&1 | tee logs/pp_ib_stepA.log; \
   echo '[DONE] stepA $(date -Is)' | tee -a logs/pp_ib_stepA.log"

sleep 5
tmux ls | grep pp_ib || true
tail -6 logs/pp_ib_stepA.log 2>/dev/null || true
