#!/usr/bin/env bash
# 在 MINER-My 根目录后台启动完整提升管线（日志 + PID）
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export GPU_ID="${GPU_ID:-0}"
export FULL_PIPELINE=1
export RUN_GRID_SEARCH="${RUN_GRID_SEARCH:-1}"
export RUN_VITL="${RUN_VITL:-1}"
export RUN_MULTI_SEED="${RUN_MULTI_SEED:-1}"
# 有 Key 时可: export RUN_GPT4V=1 RUN_CLAUDE=1

LOG=logs/full_pipeline.log
PIDFILE=logs/full_pipeline.pid

if [[ -f "$PIDFILE" ]]; then
  OLD="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "$OLD" ]] && kill -0 "$OLD" 2>/dev/null; then
    echo "Already running PID $OLD — tail -f $LOG"
    exit 0
  fi
fi

nohup env \
  PYTHONUNBUFFERED=1 \
  FULL_PIPELINE=1 \
  RUN_GRID_SEARCH="$RUN_GRID_SEARCH" \
  RUN_VITL="$RUN_VITL" \
  RUN_MULTI_SEED="$RUN_MULTI_SEED" \
  RUN_GPT4V="${RUN_GPT4V:-0}" \
  RUN_CLAUDE="${RUN_CLAUDE:-0}" \
  GPU_ID="$GPU_ID" \
  VIT_L="${VIT_L:-pretrain/clip-vit-large-patch14}" \
  bash scripts/run_all_improvements.sh >>"$LOG" 2>&1 &

echo $! >"$PIDFILE"
echo "Started PID $(cat "$PIDFILE")"
echo "Log: $ROOT/$LOG"
echo "tail -f $LOG"
