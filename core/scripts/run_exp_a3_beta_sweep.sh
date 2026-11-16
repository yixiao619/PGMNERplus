#!/usr/bin/env bash
# 用法：先等 A1/A2 出结果，再决定是否叠加 span_detection_weight。
# 默认不加辅助头（--span_detection_weight 0），A1 有效则改为 0.2。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

SPAN_W="${1:-0}"   # 如果 A2 优于 A1，传入 0.2；否则保持 0

for BP in 0.003 0.005 0.007 0.008; do
  OUT="outputs/exp_a3_bp${BP}_sw${SPAN_W}"
  mkdir -p "$OUT"
  echo ">>> 启动 beta_polar=${BP} span_w=${SPAN_W}  OUT=${OUT}"

  python -u main-mner-robert.py \
    --data_dir data/twitter2017 \
    --model_name_or_path pretrain/roberta-large \
    --tokenizer_name pretrain/roberta-large \
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
    --output_dir "$OUT" \
    --use_pp_ib \
    --beta_polar "${BP}" \
    --crf_misc_bias 1.5 \
    --crf_entity_bias 0.5 \
    --crf_o_penalty -0.3 \
    --span_detection_weight "${SPAN_W}" \
    --span_head_dropout 0.1 \
    2>&1 | tee "$OUT/train.log"

  {
    echo "# beta_polar=${BP} span_w=${SPAN_W} — $(date -Is)"
    [[ -f "$OUT/eval_results.txt" ]] && cat "$OUT/eval_results.txt"
    grep "best_score span f1" "$OUT/train.log" 2>/dev/null | tail -3 || true
  } > "$OUT/results.txt"

  echo "=== 完成 ${OUT} ==="
done
