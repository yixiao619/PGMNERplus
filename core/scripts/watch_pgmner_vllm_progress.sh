#!/usr/bin/env bash
# Live task list + progress bars for PGMNER+VLLM auto pipeline.
# Usage:
#   bash scripts/watch_pgmner_vllm_progress.sh twitter2017          # once
#   bash scripts/watch_pgmner_vllm_progress.sh twitter2017 watch  # refresh every 30s
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="${1:-twitter2017}"
MODE="${2:-once}"
DATA_DIR="$ROOT/data/${DATASET}"
RUNS="$ROOT/runs/pgmner_vllm/${DATASET}"
STATE="$ROOT/logs/pgmner_vllm_pipeline.state"
TASKS="$ROOT/logs/pgmner_vllm_TASKS.md"

expect_dev=723
expect_test=723
expect_train=3373
[[ "$DATASET" == "twitter2015" ]] && { expect_dev=1000; expect_test=3257; expect_train=4000; }

bar() {
  local n=$1 e=$2 label=$3
  local pct filled empty
  pct=$(awk -v n="$n" -v e="$e" 'BEGIN{ if (e<=0) print 0; else if (n>=e) print 100; else printf "%.1f", 100*n/e }')
  filled=$(awk -v p="$pct" 'BEGIN{ printf "%d", int(40*p/100+0.5) }')
  empty=$((40 - filled))
  printf "  %-6s [" "$label"
  printf '%0.s█' $(seq 1 "$filled" 2>/dev/null) || true
  printf '%0.s░' $(seq 1 "$empty" 2>/dev/null) || true
  printf "] %5s%%  %5d / %d\n" "$pct" "$n" "$e"
}

line_count() {
  local f="$1"
  [[ -f "$f" ]] || { echo 0; return; }
  wc -l <"$f" | tr -d '[:space:]'
}

qwen_n() {
  local f="$1"
  [[ -f "$f" ]] || { echo 0; return; }
  python3 -c "print(sum(1 for ln in open('$f') if 'Qwen3.5' in ln))" 2>/dev/null || echo 0
}

stub_n() {
  local f="$1"
  [[ -f "$f" ]] || { echo 0; return; }
  python3 -c "print(sum(1 for ln in open('$f') if '\"model\": \"stub\"' in ln))" 2>/dev/null || echo 0
}

render() {
  clear 2>/dev/null || true
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║  PGMNER + VLLM 进度  │  $DATASET  │  $(date '+%Y-%m-%d %H:%M:%S')  ║"
  echo "╚══════════════════════════════════════════════════════════════╝"
  echo ""
  phase="(unknown)"
  [[ -f "$STATE" ]] && phase=$(cat "$STATE")
  echo "当前阶段: $phase"
  echo ""
  echo "── 描述生成 (Qwen3.5-27B, GPU 2) ──"
  for sp in dev test train; do
    f="$DATA_DIR/vllm_desc/${sp}.jsonl"
    exp_var="expect_${sp}"
    exp=${!exp_var}
    n=$(line_count "$f")
    q=$(qwen_n "$f")
    s=$(stub_n "$f")
    stub=""
    [[ "$s" -gt 0 ]] 2>/dev/null && stub=" (stub残留 $s)"
    bar "$q" "$exp" "$sp"
    if [[ "$n" =~ ^[0-9]+$ && "$q" =~ ^[0-9]+$ && "$q" -lt "$n" ]]; then
      echo "         └─ 总行 $n，非 Qwen $((n - q))$stub"
    fi
  done
  echo ""
  echo "── PGMNER+VLLM 训练 (GPU 1, 目标 test span F1) ──"
  train_live=0
  [[ "$phase" == PHASE5_TRAIN* || "$phase" == PHASE6* ]] && train_live=1
  pgrep -f "main-mner-robert.py.*pgmner_vllm" >/dev/null 2>&1 && train_live=1
  for seed in 42 123 456; do
    sd="$RUNS/$seed"
    if [[ -f "$sd/eval_results.txt" ]] && grep -q "test best ckpt" "$sd/eval_results.txt" 2>/dev/null; then
      tf=$(grep -A3 "test best ckpt" "$sd/eval_results.txt" | grep "span_f1" | head -1 | awk '{print $3}')
      bar 100 100 "s$seed"
      echo "         └─ test span_f1 = $tf"
    elif [[ "$train_live" -eq 1 && -f "$sd/train.log" ]]; then
      ep=$(grep -c "Training epoch num" "$sd/train.log" 2>/dev/null || echo 0)
      bf=$(grep "best_score span f1" "$sd/train.log" 2>/dev/null | tail -1 | awk '{print $NF}')
      bar "$ep" 40 "s$seed"
      echo "         └─ epoch≈$ep/40  dev_best≈$bf"
    elif [[ -f "$sd/train.log" ]] && [[ "$train_live" -eq 0 ]]; then
      bar 0 40 "s$seed"
      echo "         └─ 旧 stub 日志残留，描述完成后将重训"
    else
      bar 0 40 "s$seed"
      echo "         └─ 等待描述完成 / 未启动"
    fi
  done
  echo ""
  echo "── GPU ──"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | \
    awk -F', ' '{printf "  GPU%s: %s util, %s / %s\n", $1, $2, $3, $4}' || true
  echo ""
  echo "── 进程 ──"
  pgrep -af "run_pgmner_vllm_auto|run_qwen35|generate_vllm_desc|run_pgmner_vllm.sh|main-mner-robert" 2>/dev/null | \
    grep -v watch | head -6 | sed 's/^/  /' || echo "  (无)"
  echo ""
  if [[ -f "$TASKS" ]]; then
    echo "── 任务清单 (logs/pgmner_vllm_TASKS.md) ──"
    tail -8 "$TASKS" | sed 's/^/  /'
  fi
  echo ""
  echo "刷新: bash scripts/watch_pgmner_vllm_progress.sh $DATASET watch"
  echo "日志: tail -f logs/pgmner_vllm_auto.log"
}

if [[ "$MODE" == "watch" ]]; then
  while true; do
    render
    sleep 30
  done
else
  render
fi
