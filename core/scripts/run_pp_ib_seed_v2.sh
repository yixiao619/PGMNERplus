#!/usr/bin/env bash
# 三 patch 后重跑崩掉的 seed（789 / 1024），输出到 pp_ib_seed*_v2
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

for SEED in 789 1024; do
  OUT="outputs/pp_ib_seed${SEED}_v2"
  if grep -q 'Eval results test best ckpt' "$OUT/eval_results.txt" 2>/dev/null; then
    echo "[SKIP] $OUT already has final eval"
    continue
  fi
  rm -rf "$OUT"
  mkdir -p "$OUT"
  echo "===== PP-IB v2 seed=$SEED -> $OUT ====="
  python main-mner-robert.py \
    --data_dir data/twitter2017 \
    --model_name_or_path pretrain/roberta-large \
    --tokenizer_name pretrain/roberta-large \
    --vit_name pretrain/clip-vit-base-patch16 \
    --use_hf_clip_vision true \
    --auto_img_dims true \
    --gpu_id 0 \
    --epoch 30 \
    --batch_size 8 \
    --max_seq_len 128 \
    --max_span_len 4 \
    --lr 3e-5 \
    --bert_lr 3e-5 \
    --patience 0 \
    --seed "$SEED" \
    --logging_steps 100 \
    --do_train true \
    --do_eval true \
    --do_predict true \
    --use_pp_ib \
    --beta_polar 0.01 \
    --delta_minus 0.2 \
    --delta_plus 0.8 \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/train.log"
  f1=$(grep 'best_score span f1' "$OUT/train.log" 2>/dev/null | tail -1 | awk '{print $NF}')
  echo "seed=${SEED} best_score_test_f1=${f1}"
done

echo ""
echo "===== v2 summary ====="
for SEED in 789 1024; do
  OUT="outputs/pp_ib_seed${SEED}_v2"
  if [[ -f "$OUT/train.log" ]]; then
    f1=$(grep 'best_score span f1' "$OUT/train.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    echo "seed=${SEED} test_f1=${f1}"
  fi
done
