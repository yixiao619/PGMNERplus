#!/usr/bin/env bash
# PGMNER-full: +PP-IB-DPP + GOT-A + GOT-A-PP + 分层学习率；Twitter-2017。PGMNER改进
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

OUT="${1:-outputs/pgmner_full_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

# 行名对应 ablation: PGMNER-full
python main-mner-robert.py \
  --data_dir data/twitter2017 \
  --output_dir "$OUT" \
  --model_name_or_path pretrain/roberta-large \
  --tokenizer_name pretrain/roberta-large \
  --vit_name pretrain/clip-vit-base-patch16 \
  --gpu_id 0 \
  --epoch 30 \
  --batch_size 8 \
  --max_seq_len 128 \
  --max_span_len 4 \
  --patience 0 \
  --seed 42 \
  --logging_steps 100 \
  --do_train true \
  --do_eval true \
  --do_predict true \
  --txt_global_dim 1024 \
  --txt_local_dim 1024 \
  --img_global_dim 768 \
  --img_local_dim 768 \
  --hidden_dim 1024 \
  --use_tiered_lr \
  --text_encoder_lr 2e-6 \
  --vision_encoder_lr 2e-6 \
  --main_lr 1e-5 \
  --bert_lr 2e-6 \
  --lr 1e-5 \
  --warmup_ratio 0.1 \
  --use_pp_ib \
  --use_pp_ib_dpp \
  --dpp_warmup_epochs 5 \
  --beta_polar 0.005 \
  --beta_schedule static \
  --beta_polar_warmup_epochs 3 \
  --use_got_cac \
  --lambda_gw 0.2 \
  --got_outer_iter 2 \
  --got_lambda_warmup_epochs 3 \
  --use_got_pp \
  --lambda_got_pp 0.05 \
  --got_pp_top_k 3 \
  --got_pp_epsilon 0.01 \
  --ot_alpha 0.05 \
  2>&1 | tee "$OUT/train.log"

{
  echo "# PGMNER-full — $(date -Is)"
  if [[ -f "$OUT/eval_results.txt" ]]; then
    cat "$OUT/eval_results.txt"
  fi
  echo ""
  grep "best_score span f1" "$OUT/train.log" 2>/dev/null | tail -5 || true
} >"$OUT/results_summary.txt" || true

echo "== Done. OUT=$OUT =="
