#!/usr/bin/env bash
# 消融：仅改进一 +PP-IB-DPP（动态极化），不开 GOT-A / GOT-A-PP / 分层学习率。
# 行名：+PP-IB-DPP（或相对 baseline 记 +DPP 增量需另跑纯 baseline）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

OUT="${1:-outputs/exp_pp_ib_dpp_only_$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${2:-0}"
mkdir -p "$OUT"

# -u: 不缓冲 stdout/stderr，避免 pipe|tee 时长时间看不到 “Running training / Epoch”
python -u main-mner-robert.py \
  --data_dir data/twitter2017 \
  --output_dir "$OUT" \
  --model_name_or_path pretrain/roberta-large \
  --tokenizer_name pretrain/roberta-large \
  --vit_name pretrain/clip-vit-base-patch16 \
  --gpu_id "$GPU_ID" \
  --epoch 30 \
  --batch_size 8 \
  --max_seq_len 128 \
  --max_span_len 4 \
  --lr 3e-5 \
  --bert_lr 3e-5 \
  --warmup_ratio 0.1 \
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
  --use_pp_ib \
  --use_pp_ib_dpp \
  --dpp_warmup_epochs 5 \
  --beta_polar 0.005 \
  --beta_schedule static \
  --beta_polar_warmup_epochs 3 \
  --delta_minus 0.2 \
  --delta_plus 0.8 \
  2>&1 | tee "$OUT/train.log"

{
  echo "# +PP-IB-DPP only — $(date -Is)"
  if [[ -f "$OUT/eval_results.txt" ]]; then cat "$OUT/eval_results.txt"; fi
  echo ""
  grep "best_score span f1" "$OUT/train.log" 2>/dev/null | tail -3 || true
} >"$OUT/summary.txt" || true

echo "== Done OUT=$OUT GPU=$GPU_ID =="
