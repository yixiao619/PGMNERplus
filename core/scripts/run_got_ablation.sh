#!/usr/bin/env bash
# GOT-A / Partial OT 消融（main-mner-robert.py）；输出目录默认 outputs/got_ablation_new
# 用法: OUT=/path/to/out bash scripts/run_got_ablation.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${OUT:-$ROOT/outputs/got_ablation_new}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1

BASE=(
  --data_dir data/twitter2017
  --model_name_or_path pretrain/roberta-large
  --tokenizer_name pretrain/roberta-large
  --vit_name pretrain/clip-vit-base-patch16
  --gpu_id 0
  --epoch 30
  --batch_size 8
  --max_seq_len 128
  --max_span_len 4
  --lr 3e-5
  --bert_lr 3e-5
  --patience 0
  --seed 42
  --logging_steps 100
  --do_train true
  --do_eval true
  --do_predict true
  --use_pp_ib
  --beta_polar 0.01
  --ot_alpha 0.05
  --ot_eps 0.1
  --ot_sinkhorn_iter 30
  --ot_lambda_align 1.0
)

echo "=== A: PP-IB only (no GOT / no SemanticOT path) ==="
python main-mner-robert.py "${BASE[@]}" \
  --output_dir "$OUT/A_ppib_only" \
  2>&1 | tee "$OUT/A_ppib_only.log"

echo "=== B: use_got + node_sinkhorn (full-marginal entropic OT, no GW) ==="
python main-mner-robert.py "${BASE[@]}" \
  --use_got --got_ot_backend node_sinkhorn \
  --output_dir "$OUT/B_node_sinkhorn" \
  2>&1 | tee "$OUT/B_node_sinkhorn.log"

echo "=== C: partial_only (small eps, learnable s) ==="
python main-mner-robert.py "${BASE[@]}" \
  --use_got --got_ot_backend partial_only \
  --partial_mass_init 0.6 \
  --dust_cost 0.5 \
  --output_dir "$OUT/C_partial_only" \
  2>&1 | tee "$OUT/C_partial_only.log"

echo "=== D: entropy_only (NodeSinkhorn / full mass + entropy) ==="
python main-mner-robert.py "${BASE[@]}" \
  --use_got --got_ot_backend entropy_only \
  --output_dir "$OUT/D_entropy_only" \
  2>&1 | tee "$OUT/D_entropy_only.log"

echo "=== E: partial_sinkhorn (proposed Partial OT + entropy) ==="
python main-mner-robert.py "${BASE[@]}" \
  --use_got --got_ot_backend partial_sinkhorn \
  --partial_mass_init 0.6 \
  --dust_cost 0.5 \
  --output_dir "$OUT/E_partial_sinkhorn" \
  2>&1 | tee "$OUT/E_partial_sinkhorn.log"

echo "=== F: gw (semantic + Gromov–Wasserstein, legacy GOT) ==="
python main-mner-robert.py "${BASE[@]}" \
  --use_got --got_ot_backend gw \
  --got_lambda_sem 0.5 \
  --got_struct_mode full \
  --output_dir "$OUT/F_gw_original" \
  2>&1 | tee "$OUT/F_gw_original.log"

SUMMARY="$OUT/IMPORTANT_RESULTS.txt"
{
  echo "══════════════════════════════════════════════════════════════════════"
  echo "GOT / Partial OT 消融汇总  generated_at=$(date -Is)"
  echo "ROOT=$ROOT  OUT=$OUT"
  echo "══════════════════════════════════════════════════════════════════════"
  for key in A_ppib_only B_node_sinkhorn C_partial_only D_entropy_only E_partial_sinkhorn F_gw_original; do
    log="$OUT/${key}.log"
    sub="$OUT/${key}"
    echo ""
    echo "── $key ──"
    if [[ -f "$log" ]]; then
      echo "# dev 最佳 F1（grep）"
      grep -E 'best_score span f1:|best score f1:' "$log" | tail -8 || true
      echo "# test / 预测相关（若存在）"
      grep -iE 'test|span_f1|eval_results' "$log" | tail -15 || true
    else
      echo "(无 $log)"
    fi
    if [[ -f "$sub/eval_results.txt" ]]; then
      echo "# $sub/eval_results.txt"
      cat "$sub/eval_results.txt"
    fi
  done
  echo ""
  echo "各实验完整日志: $OUT/*.log"
} | tee "$SUMMARY"

echo "完成。汇总: $SUMMARY  日志与 checkpoint: $OUT/"
