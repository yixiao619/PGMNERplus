#!/usr/bin/env bash
# PART D2：lambda_gw 扫描（GOT-CAC only，对齐 exp3 除 lambda 外；此处 outer_iter=2）
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
  --do_eval true --do_predict true
  --use_got_cac
  --got_outer_iter 2
)

gw_to_name() {
  python3 -c "import sys; x=float(sys.argv[1]); print('sens_gw_%s' % str(x).replace('.','p'))" "$1"
}

for gw in 0.01 0.05 0.1 0.15 0.2 0.3 0.5 0.8 1.0; do
  sub=$(gw_to_name "$gw")
  out="outputs/$sub"
  if is_job_complete "$out"; then
    read -r tf df <<< "$(read_out_f1 "$out")"
    echo "=== $sub 已完成，跳过 | test_f1=${tf} | dev_f1=${df} ==="
    continue
  fi
  mkdir -p "$out"
  echo "======== $sub lambda_gw=$gw ========"
  "${COMMON[@]}" --lambda_gw "$gw" --output_dir "$out" 2>&1 | tee "$out/train.log"
  read -r tf df <<< "$(extract_tf_df "$out/eval_results.txt")"
  echo "=== $sub done | test_f1=${tf} | dev_f1=${df} ==="
done

python3 "$ROOT/scripts/merge_sensitivity_summary.py"
