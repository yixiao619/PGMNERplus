#!/usr/bin/env bash
# Twitter-2017：对比实验 (1) baseline (2) PP-IB (3) GOT-A CAC (4) 二者
# 顺序执行（单 GPU 避免 OOM）。用法（后台、关终端可继续）：
#   cd MINER-My && nohup bash scripts/run_twitter2017_ablation_1to4.sh >> checkpoints/ablation_chain/nohup_master.log 2>&1 &
# 说明：整机断电/关机后进程无法继续，需未关机的服务器或 tmux/systemd。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CHAIN_DIR="checkpoints/ablation_chain"
mkdir -p "$CHAIN_DIR"
SUMMARY="$CHAIN_DIR/results_summary.txt"
VLOG="/home/jyx/yixiao/VLOG_miner_pp_got_ablation.md"

export PYTHONUNBUFFERED=1

# 需要联网拉模型时可打开：
# export HF_ENDPOINT=https://hf-mirror.com

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
)

run_exp() {
  local tag="$1"
  local out_dir="$2"
  shift 2
  local logf="$CHAIN_DIR/train_${tag}.log"
  echo "========== $(date -Is) START $tag -> $out_dir ==========" | tee -a "$SUMMARY"
  mkdir -p "$out_dir"
  "${COMMON[@]}" \
    --output_dir "$out_dir" \
    "$@" \
    >"$logf" 2>&1

  echo "========== $(date -Is) DONE $tag ==========" | tee -a "$SUMMARY"
  # 训练内最优 test F1（与脚本逻辑一致）
  if grep -q "best_score span f1" "$logf"; then
    grep "best_score span f1" "$logf" | tail -1 | tee -a "$SUMMARY"
  fi
  # 正式 dev / test（best ckpt）
  local er="$out_dir/eval_results.txt"
  if [[ -f "$er" ]]; then
    echo "--- eval_results.txt (dev/test best ckpt 片段) ---" | tee -a "$SUMMARY"
    grep -A4 "Eval results dev best ckpt" "$er" | tail -5 | tee -a "$SUMMARY" || true
    grep -A4 "Eval results test best ckpt" "$er" | tail -5 | tee -a "$SUMMARY" || true
  fi
  echo "" | tee -a "$SUMMARY"
}

echo "Ablation chain started $(date -Is)" | tee "$SUMMARY"

run_exp "1_baseline" "checkpoints/twitter2017-ablation-1-baseline"

run_exp "2_pp_ib" "checkpoints/twitter2017-ablation-2-ppib" --use_pp_ib --beta_polar 0.01

run_exp "3_got_cac" "checkpoints/twitter2017-ablation-3-gotcac" --use_got_cac --lambda_gw 0.3 --got_outer_iter 2

run_exp "4_both" "checkpoints/twitter2017-ablation-4-both" --use_pp_ib --beta_polar 0.01 --use_got_cac --lambda_gw 0.3 --got_outer_iter 2

echo "All four finished $(date -Is)" | tee -a "$SUMMARY"

# 追加到 vlog（若存在）
if [[ -f "$VLOG" ]]; then
  {
    echo ""
    echo "## 实验结果汇总（自动生成，$(date -Is)）"
    echo ""
    echo '```'
    cat "$SUMMARY"
    echo '```'
  } >>"$VLOG"
fi
