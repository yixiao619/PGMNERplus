#!/usr/bin/env bash
# PART B2：GOT-CAC 图结构（Twitter-2017）；其余对齐 exp3 变体：lambda_gw=0.1, got_outer_iter=2
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
# shellcheck source=/dev/null
source "$ROOT/scripts/pgmner_job_util.sh"

REF_DEFAULT_F1=0.88832
WARN_THRESH=0.003

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
  --use_got_cac
  --lambda_gw 0.1
  --got_outer_iter 2
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

run_one got_k_default outputs/got_k_default --got_graph_mode full
read -r TF0 _ <<< "$(read_out_f1 outputs/got_k_default)"
if [[ "$TF0" != "NA" ]]; then
  python3 - <<PY || true
import math
ref = float("${REF_DEFAULT_F1}")
got = float("${TF0}")
if abs(got - ref) > float("${WARN_THRESH}"):
    print(f"WARNING: got_k_default test_f1={got:.5f} vs ref {ref:.5f} (|diff|>{float('${WARN_THRESH}')})")
else:
    print(f"OK got_k_default close to ref (|diff|={abs(got-ref):.5f})")
PY
fi

run_one got_k3 outputs/got_k3 --got_graph_mode knn --got_graph_k 3
run_one got_k5 outputs/got_k5 --got_graph_mode knn --got_graph_k 5
run_one got_k7 outputs/got_k7 --got_graph_mode knn --got_graph_k 7
run_one got_thr03 outputs/got_thr03 --got_graph_mode threshold --got_graph_threshold 0.3
run_one got_thr05 outputs/got_thr05 --got_graph_mode threshold --got_graph_threshold 0.5

echo "PART B2 全部完成。"
