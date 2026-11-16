#!/usr/bin/env bash
# PP-IB + GOT-CAC 超参网格：顺序执行 6 组，其余与 run_exp_both.sh 一致
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

COMMON=(
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
  --patience 0  --seed 42
  --logging_steps 100
  --do_train true
  --do_eval true
  --do_predict true
  --use_pp_ib
  --delta_minus 0.2
  --delta_plus 0.8
  --use_got_cac
  --got_outer_iter 2
)

run_one() {
  local out=$1
  shift
  mkdir -p "$out"
  python main-mner-robert.py "${COMMON[@]}" "$@" --output_dir "$out" \
    2>&1 | tee "$out/train.log"
  {
    echo "# results.txt — $(date -Is)"
    if [[ -f "$out/eval_results.txt" ]]; then
      cat "$out/eval_results.txt"
    fi
    echo ""
    grep "best_score span f1" "$out/train.log" 2>/dev/null | tail -3 || true
  } >"$out/results.txt" || true
  echo "=== $(basename "$out") done ==="
}

run_one outputs/grid_1 --beta_polar 0.005 --lambda_gw 0.3
run_one outputs/grid_2 --beta_polar 0.005 --lambda_gw 0.1
run_one outputs/grid_3 --beta_polar 0.001 --lambda_gw 0.3
run_one outputs/grid_4 --beta_polar 0.001 --lambda_gw 0.1
run_one outputs/grid_5 --beta_polar 0.01 --lambda_gw 0.1
run_one outputs/grid_6 --beta_polar 0.005 --lambda_gw 0.2 --beta_polar_warmup_epochs 3

echo "=== run_both_grid.sh: all 6 jobs finished ==="
