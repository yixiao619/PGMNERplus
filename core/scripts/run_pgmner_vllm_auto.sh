#!/usr/bin/env bash
# Full auto pipeline: Qwen3.5 desc (dev resume → test/train fresh) → PGMNER+VLLM train (F1).
# Usage:
#   nohup bash scripts/run_pgmner_vllm_auto.sh twitter2017 > logs/pgmner_vllm_auto_nohup.log 2>&1 &
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

DATASET="${1:-twitter2017}"
GPU_GEN="${GPU_GEN:-2}"
GPU_TRAIN="${GPU_TRAIN:-1}"
STATE="$ROOT/logs/pgmner_vllm_pipeline.state"
TASKS="$ROOT/logs/pgmner_vllm_TASKS.md"
LOG="$ROOT/logs/pgmner_vllm_auto.log"

exec >>"$LOG" 2>&1

expect_dev=723
expect_test=723
expect_train=3373
[[ "$DATASET" == "twitter2015" ]] && { expect_dev=1000; expect_test=3257; expect_train=4000; }

write_state() { echo "$1" >"$STATE"; echo "$(date -Is) $1" >>"$LOG"; }
mark_task() { echo "- [$2] **$1** — $3 ($(date '+%H:%M:%S'))" >>"$TASKS"; }

mkdir -p logs
if [[ ! -f "$TASKS" ]]; then
  cat >"$TASKS" <<EOF
# PGMNER+VLLM 任务清单 ($DATASET)

| # | 任务 | 状态 |
|---|------|------|
| 1 | dev 描述 (Qwen3.5, 续跑) | 待开始 |
| 2 | test 描述 (Qwen3.5, 覆盖 stub) | 待开始 |
| 3 | train 描述 (Qwen3.5, 覆盖 stub) | 待开始 |
| 4 | PGMNER+VLLM 训练 seed 42/123/456 | 待开始 |
| 5 | test F1 汇总 + LaTeX | 待开始 |

实时进度: \`bash scripts/watch_pgmner_vllm_progress.sh $DATASET\`

EOF
fi

qwen_lines() {
  local f="$1"
  [[ -f "$f" ]] || { echo 0; return; }
  grep -c 'Qwen3.5' "$f" 2>/dev/null || echo 0
}

wait_split_done() {
  local split=$1 expect=$2 f="$ROOT/data/${DATASET}/vllm_desc/${split}.jsonl"
  while true; do
    n=$(wc -l <"$f" 2>/dev/null || echo 0)
    q=$(qwen_lines "$f")
    if [[ "$n" -ge "$expect" ]] && [[ "$q" -ge "$expect" ]]; then
      return 0
    fi
    sleep 60
  done
}

echo "======== AUTO PIPELINE START $(date -Is) dataset=$DATASET ========"

# --- Phase 1: dev (resume, no FRESH) ---
write_state "PHASE1_DEV_GEN"
mark_task "dev Qwen3.5 描述" "进行中" "GPU $GPU_GEN 续跑"
CUDA_VISIBLE_DEVICES="$GPU_GEN" FRESH=0 bash scripts/run_qwen35_vllm_desc.sh "$DATASET" dev
mark_task "dev Qwen3.5 描述" "完成" "lines=$(wc -l <data/${DATASET}/vllm_desc/dev.jsonl)"

# --- Phase 2: test (FRESH, replace stub) ---
write_state "PHASE2_TEST_GEN"
mark_task "test Qwen3.5 描述" "进行中" "GPU $GPU_GEN FRESH=1"
CUDA_VISIBLE_DEVICES="$GPU_GEN" FRESH=1 bash scripts/run_qwen35_vllm_desc.sh "$DATASET" test
mark_task "test Qwen3.5 描述" "完成" ""

# --- Phase 3: train (FRESH) ---
write_state "PHASE3_TRAIN_GEN"
mark_task "train Qwen3.5 描述" "进行中" "GPU $GPU_GEN FRESH=1 (~5-6h)"
CUDA_VISIBLE_DEVICES="$GPU_GEN" FRESH=1 bash scripts/run_qwen35_vllm_desc.sh "$DATASET" train
mark_task "train Qwen3.5 描述" "完成" ""

# --- Phase 4: validate cache ---
write_state "PHASE4_VALIDATE"
for sp in dev test train; do
  f="data/${DATASET}/vllm_desc/${sp}.jsonl"
  n=$(wc -l <"$f")
  q=$(qwen_lines "$f")
  exp_var="expect_${sp}"
  exp=${!exp_var}
  if [[ "$q" -lt "$exp" ]]; then
    echo "ERROR: $sp only $q/$exp Qwen lines" >&2
    write_state "FAILED_VALIDATE_${sp}"
    exit 1
  fi
done
mark_task "描述缓存校验" "完成" "dev/test/train 均为 Qwen3.5"

# --- Phase 5: PGMNER train ---
write_state "PHASE5_TRAIN"
mark_task "PGMNER+VLLM 训练" "进行中" "GPU $GPU_TRAIN seeds 42,123,456"
SKIP_GEN=1 VLLM_BACKEND=qwen35 USE_VLLM_DESC=1 CUDA_VISIBLE_DEVICES="$GPU_TRAIN" \
  bash scripts/run_pgmner_vllm.sh "$DATASET"
mark_task "PGMNER+VLLM 训练" "完成" ""

write_state "PHASE6_DONE"
mark_task "LaTeX / 结果汇总" "完成" "runs/pgmner_vllm/${DATASET}/"
echo "======== AUTO PIPELINE DONE $(date -Is) ========"
