#!/usr/bin/env bash
# PART D1：beta_polar 扫描（PP-IB only，对齐 exp2 除 beta 外超参）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
# shellcheck source=/dev/null
source "$ROOT/scripts/pgmner_job_util.sh"

COMMON=(
  python main-mner-robert.py
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
)

while IFS=$'\t' read -r b sub; do
  [[ -z "${b:-}" || "${b:0:1}" == "#" ]] && continue
  out="outputs/$sub"
  if is_job_complete "$out"; then
    read -r tf df <<< "$(read_out_f1 "$out")"
    echo "=== $sub 已完成，跳过 | test_f1=${tf} | dev_f1=${df} ==="
    continue
  fi
  mkdir -p "$out"
  echo "======== $sub beta_polar=$b ========"
  "${COMMON[@]}" --beta_polar "$b" --output_dir "$out" 2>&1 | tee "$out/train.log"
  read -r tf df <<< "$(extract_tf_df "$out/eval_results.txt")"
  echo "=== $sub done | test_f1=${tf} | dev_f1=${df} ==="
done <<'TAB'
0.0001	sens_beta_1e-4
0.0005	sens_beta_5e-4
0.001	sens_beta_1e-3
0.003	sens_beta_3e-3
0.005	sens_beta_5e-3
0.008	sens_beta_8e-3
0.01	sens_beta_1e-2
0.02	sens_beta_2e-2
0.05	sens_beta_5e-2
TAB

python3 "$ROOT/scripts/merge_sensitivity_summary.py"
