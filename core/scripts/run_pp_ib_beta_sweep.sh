#!/usr/bin/env bash
# PP-IB β_polar 网格（固定 seed=42）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

BETAS=(0.001 0.003 0.005 0.01 0.02 0.05)
for B in "${BETAS[@]}"; do
  BTAG=$(echo "$B" | sed 's/\.//g')
  OUT="outputs/pp_ib_beta${BTAG}"
  if [[ -f "$OUT/eval_results.txt" ]]; then
    echo "[SKIP] $OUT"
    continue
  fi
  mkdir -p "$OUT"
  echo "===== Running PP-IB beta_polar=$B ====="
  python main-mner-robert.py \
    --data_dir data/twitter2017 \
    --model_name_or_path pretrain/roberta-large \
    --tokenizer_name pretrain/roberta-large \
    --vit_name pretrain/clip-vit-base-patch16 \
    --use_hf_clip_vision true \
    --auto_img_dims true \
    --gpu_id 0 --epoch 30 --batch_size 8 \
    --max_seq_len 128 --max_span_len 4 \
    --lr 3e-5 --bert_lr 3e-5 \
    --patience 0 --seed 42 \
    --logging_steps 100 \
    --do_train true --do_eval true --do_predict true \
    --use_pp_ib \
    --beta_polar "$B" \
    --delta_minus 0.2 --delta_plus 0.8 \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/train.log"
done

echo ""
echo "===== beta_polar sweep summary ====="
for B in "${BETAS[@]}"; do
  BTAG=$(echo "$B" | sed 's/\.//g')
  OUT="outputs/pp_ib_beta${BTAG}"
  if [[ -f "$OUT/train.log" ]]; then
    f1=$(grep "best_score span f1" "$OUT/train.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    echo "beta_polar=${B} best_score_test_f1=${f1}"
  fi
done
