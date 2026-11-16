#!/usr/bin/env bash
# PP-IB 多 seed 跑，估计随机性方差（ViT-B/16 + HF CLIP，与 task6 对齐）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

SEEDS=(42 123 456 789 1024)
FIXED_B16="outputs/pp_ib_fixed_b16"
for SEED in "${SEEDS[@]}"; do
  OUT="outputs/pp_ib_seed${SEED}"
  if [[ -f "$OUT/eval_results.txt" ]]; then
    echo "[SKIP] $OUT 已有结果"
    continue
  fi
  # seed=42 与 task6 的 pp_ib_fixed_b16 配置一致，直接复用省一轮训练
  if [[ "$SEED" == "42" ]] && [[ -f "$FIXED_B16/eval_results.txt" ]]; then
    echo "[LINK] seed=42 → 从 $FIXED_B16 复制到 $OUT"
    mkdir -p "$OUT"
    cp -a "$FIXED_B16/." "$OUT/"
    continue
  fi
  mkdir -p "$OUT"
  echo "===== Running PP-IB seed=$SEED ====="
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
done

echo ""
echo "===== Seed sweep summary ====="
for SEED in "${SEEDS[@]}"; do
  OUT="outputs/pp_ib_seed${SEED}"
  if [[ -f "$OUT/train.log" ]]; then
    f1=$(grep "best_score span f1" "$OUT/train.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    echo "seed=${SEED} best_score_test_f1=${f1}"
  fi
done
