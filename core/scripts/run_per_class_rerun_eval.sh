#!/usr/bin/env bash
# PART C2：对已训练目录仅评估（须与训练时开关一致），并生成 outputs/per_class_summary.txt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
# shellcheck source=/dev/null
source "$ROOT/scripts/pgmner_job_util.sh"

BASE=(
  python main-mner-robert.py
  --model_name_or_path pretrain/roberta-large
  --tokenizer_name pretrain/roberta-large
  --vit_name pretrain/clip-vit-base-patch16
  --gpu_id 0
  --max_seq_len 128
  --max_span_len 4
  --eval_only
)

run_eval() {
  local data_dir=$1
  local out=$2
  shift 2
  [[ -f "$out/best_model.pth" ]] || { echo "SKIP missing $out/best_model.pth"; return; }
  # 幂等：eval_results 已含 per_class_f1（或仅有 results.txt 且无 eval 时继续跑以生成 eval）
  if [[ -f "$out/eval_results.txt" ]] && grep -q "per_class_f1:" "$out/eval_results.txt"; then
    read -r tf df <<< "$(read_out_f1 "$out")"
    echo "=== per_class $(basename "$out") 已完成，跳过 | test_f1=${tf} | dev_f1=${df} ==="
    return
  fi
  echo "---- eval_only $out ----"
  "${BASE[@]}" --data_dir "$data_dir" --output_dir "$out" "$@"
}

# Twitter-2017（exp3 与仓库 run_exp_got_cac 一致 lambda_gw=0.3；grid_6 见 run_both_grid.sh）
run_eval data/twitter2017 outputs/exp1_baseline
run_eval data/twitter2017 outputs/exp2_pp_ib \
  --use_pp_ib --beta_polar 0.005 --delta_minus 0.2 --delta_plus 0.8
run_eval data/twitter2017 outputs/exp3_got_cac \
  --use_got_cac --lambda_gw 0.3 --got_outer_iter 2
run_eval data/twitter2017 outputs/grid_6 \
  --use_pp_ib --beta_polar 0.005 --delta_minus 0.2 --delta_plus 0.8 \
  --use_got_cac --lambda_gw 0.2 --got_outer_iter 2 --beta_polar_warmup_epochs 3

# Twitter-2015（见 scripts/run_twitter2015.sh）
run_eval data/twitter2015 outputs/t15_exp1_baseline
run_eval data/twitter2015 outputs/t15_exp2_pp_ib \
  --use_pp_ib --beta_polar 0.005 --delta_minus 0.2 --delta_plus 0.8
run_eval data/twitter2015 outputs/t15_exp3_got_cac \
  --use_got_cac --lambda_gw 0.2 --got_outer_iter 2
run_eval data/twitter2015 outputs/t15_exp4_both \
  --use_pp_ib --beta_polar 0.005 --delta_minus 0.2 --delta_plus 0.8 \
  --use_got_cac --lambda_gw 0.2 --got_outer_iter 2 --beta_polar_warmup_epochs 3

python3 "$ROOT/scripts/collect_per_class_summary.py"
