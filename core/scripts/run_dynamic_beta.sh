#!/usr/bin/env bash
# PART A2：Twitter-2017 动态 beta_polar 调度（除 schedule 外对齐 exp2_pp_ib：beta_polar=0.005）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
# shellcheck source=/dev/null
source "$ROOT/scripts/pgmner_job_util.sh"

# 历史 exp2 参考，仅用于 WARNING，不中断
HISTORICAL_REF_F1=0.90551
THRESH=0.005

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
  --beta_polar 0.005
  --delta_minus 0.2
  --delta_plus 0.8
)

run_one() {
  local tag=$1
  local out=$2
  shift 2
  if is_job_complete "$out"; then
    read -r tf df <<< "$(read_out_f1 "$out")"
    echo "=== ${tag} 已完成，跳过 | test_f1=${tf} | dev_f1=${df} ==="
    return
  fi
  mkdir -p "$out"
  "${COMMON[@]}" "$@" --output_dir "$out" 2>&1 | tee "$out/train.log"
  read -r tf df <<< "$(extract_tf_df "$out/eval_results.txt")"
  echo "=== ${tag} done | test_f1=${tf} | dev_f1=${df} ==="
}

report_vs_static_baseline() {
  local tag=$1
  local tf=$2
  local base=$3
  if [[ "$tf" == "NA" ]]; then
    echo "  (vs baseline) ${tag}: test_f1=NA，跳过对比"
    return
  fi
  python3 - "$HISTORICAL_REF_F1" "$base" "$tag" "$tf" <<'PY'
import sys
hist = float(sys.argv[1])
base = float(sys.argv[2])
tag = sys.argv[3]
tf = float(sys.argv[4])
print(f"  (vs dyn_static 本次基准 {base:.5f}) {tag}: Δtest_f1={tf - base:+.5f}")
print(f"  (vs 历史 exp2 参考 {hist:.5f}) {tag}: Δtest_f1={tf - hist:+.5f}")
PY
}

# 1) dyn_static
run_one dyn_static outputs/dyn_static --beta_schedule static
read -r STATIC_BASELINE _ <<< "$(read_out_f1 outputs/dyn_static)"
if [[ "$STATIC_BASELINE" == "NA" ]]; then
  echo "ERROR: dyn_static 无有效 test F1，无法作为后续基准"
  exit 1
fi

python3 - "$HISTORICAL_REF_F1" "$STATIC_BASELINE" "$THRESH" <<'PY'
import sys
hist = float(sys.argv[1])
got = float(sys.argv[2])
th = float(sys.argv[3])
d = abs(got - hist)
if d > th:
    print(
        f"WARNING: dyn_static F1 偏差 |{got:.5f}-{hist:.5f}|={d:.4f} > {th}，"
        f"以本次 dyn_static={got:.5f} 为基准继续",
        file=sys.stderr,
    )
else:
    print(f"OK dyn_static 与历史参考接近 |diff|={d:.5f} <= {th}")
PY

run_one dyn_linear outputs/dyn_linear --beta_schedule linear
read -r tf _ <<< "$(read_out_f1 outputs/dyn_linear)"
report_vs_static_baseline "dyn_linear" "$tf" "$STATIC_BASELINE"

run_one dyn_cosine outputs/dyn_cosine --beta_schedule cosine
read -r tf _ <<< "$(read_out_f1 outputs/dyn_cosine)"
report_vs_static_baseline "dyn_cosine" "$tf" "$STATIC_BASELINE"

run_one dyn_step outputs/dyn_step --beta_schedule step
read -r tf _ <<< "$(read_out_f1 outputs/dyn_step)"
report_vs_static_baseline "dyn_step" "$tf" "$STATIC_BASELINE"

run_one dyn_cosine_both outputs/dyn_cosine_both \
  --beta_schedule cosine \
  --use_got_cac \
  --lambda_gw 0.2 \
  --beta_polar_warmup_epochs 0 \
  --got_outer_iter 2
read -r tf _ <<< "$(read_out_f1 outputs/dyn_cosine_both)"
report_vs_static_baseline "dyn_cosine_both" "$tf" "$STATIC_BASELINE"

echo "PART A2 全部完成。"
