#!/usr/bin/env bash
# Twitter-2015 四组对照（与 2017 消融一致；exp4 超参 = grid_6）
# 参数名以 main-mner-robert.py 为准：--data_dir（无 --dataset）
# 注意：--use_pp_ib / --use_got_cac 为 store_true，关闭时请勿传参（不能写 false）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

# 可通过环境变量覆盖（例如数据在 /data/jyx 时：T15_DATA_DIR=/data/jyx/twitter2015）
T15_DATA_DIR="${T15_DATA_DIR:-data/twitter2015}"

COMMON=(
  --data_dir "$T15_DATA_DIR"
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
)

extract_f1() {
  local out="$1"
  local tf df
  tf=$(grep -A5 "Eval results test best ckpt" "$out/results.txt" 2>/dev/null | grep "span_f1" | head -1 | sed 's/.*= *//;s/ *$//')
  df=$(grep -A5 "Eval results dev best ckpt" "$out/results.txt" 2>/dev/null | grep "span_f1" | head -1 | sed 's/.*= *//;s/ *$//')
  tf=${tf:-NA}
  df=${df:-NA}
  echo "$tf" "$df"
}

run_one() {
  local tag=$1
  local out=$2
  shift 2
  mkdir -p "$out"
  python main-mner-robert.py "${COMMON[@]}" "$@" --output_dir "$out" 2>&1 | tee "$out/train.log"
  {
    echo "# results.txt — $(date -Is)"
    if [[ -f "$out/eval_results.txt" ]]; then
      cat "$out/eval_results.txt"
    fi
    echo ""
    grep "best_score span f1" "$out/train.log" 2>/dev/null | tail -3 || true
  } >"$out/results.txt" || true
  read -r tf df <<< "$(extract_f1 "$out")"
  echo "=== ${tag} done | test_f1=${tf} | dev_f1=${df} ==="
  echo "${tag}"$'\t'"${tf}"$'\t'"${df}" >>"$ROWS"
}

SUMMARY="${ROOT}/outputs/t15_summary.txt"
ROWS="${ROOT}/outputs/t15_summary_rows.tsv"
mkdir -p "${ROOT}/outputs"
{
  echo "# Twitter-2015 四组对照 — $(date -Is)"
  echo "# data_dir=${T15_DATA_DIR}"
  echo ""
} >"$SUMMARY"
: >"$ROWS"

# exp1：baseline（不传 use_pp_ib / use_got_cac）
run_one t15_exp1_baseline outputs/t15_exp1_baseline

# exp2：仅 PP-IB（grid_6 的 beta_polar）
run_one t15_exp2_pp_ib outputs/t15_exp2_pp_ib \
  --use_pp_ib \
  --beta_polar 0.005 \
  --delta_minus 0.2 \
  --delta_plus 0.8

# exp3：仅 GOT-CAC（grid_6 的 lambda_gw）
run_one t15_exp3_got_cac outputs/t15_exp3_got_cac \
  --use_got_cac \
  --lambda_gw 0.2 \
  --got_outer_iter 2

# exp4：双开 + grid_6 全套
run_one t15_exp4_both outputs/t15_exp4_both \
  --use_pp_ib \
  --beta_polar 0.005 \
  --delta_minus 0.2 \
  --delta_plus 0.8 \
  --use_got_cac \
  --lambda_gw 0.2 \
  --got_outer_iter 2 \
  --beta_polar_warmup_epochs 3

{
  echo ""
  echo "## 汇总表"
  echo ""
  echo "| 组别 | test F1 | dev F1 |"
  echo "|------|---------|--------|"
  while IFS=$'\t' read -r tag tf df; do
    [[ -z "${tag:-}" ]] && continue
    echo "| ${tag} | ${tf} | ${df} |"
  done <"$ROWS"
  echo ""
  echo "## TSV"
  echo -e "run\ttest_f1\tdev_f1"
  cat "$ROWS"
} >>"$SUMMARY"

echo "=== 全部完成，见 $SUMMARY ==="
cat "$SUMMARY"
