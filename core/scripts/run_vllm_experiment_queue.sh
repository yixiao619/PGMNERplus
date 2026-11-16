#!/usr/bin/env bash
# VLLM-only serial queue on one GPU. Skips finished runs; retries once on failure.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

GPU_FILTER="${1:?gpu id filter (0|1|2)}"
QUEUE="${QUEUE:-$ROOT/scripts/vllm_experiment_queue.tsv}"
LOG_DIR="$ROOT/logs/vllm_sweep"
SUMMARY="$LOG_DIR/summary_$(date +%Y%m%d).md"
LOCK="$LOG_DIR/gpu${GPU_FILTER}.lock"
PHASE="${PHASE:-}"

mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_FILTER}"
export SKIP_GEN=1
export VLLM_BACKEND=qwen35
export USE_VLLM_DESC=1
export FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
export RUNS_BASE="${RUNS_BASE:-/data/jyx/MINER-My/runs/pgmner_vllm}"
mkdir -p "$RUNS_BASE"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date -Is)] another queue on GPU $GPU_FILTER; exit"
  exit 0
fi

log() { echo "[$(date -Is)] [GPU$GPU_FILTER] $*" | tee -a "$LOG_DIR/queue_gpu${GPU_FILTER}${PHASE}.log"; }

run_job() {
  local tag="$1" dust="$2" mass="$3" seeds="$4" patience="$5" desc_max="$6" max_seq="$7"
  local lr="$8" ema="$9" beta="${10:-}" ot="${11:-}" batch="${12:-}" epochs="${13:-}" note="${14:-}"

  export RUN_TAG="$tag"
  export USE_VLLM_DESC=1
  export DUST_COST="$dust"
  export PARTIAL_MASS="$mass"
  export SEEDS="$seeds"
  export PATIENCE="$patience"
  export DESC_MAX_WORDS="${desc_max:-60}"
  [[ "$DESC_MAX_WORDS" == "-" ]] && export DESC_MAX_WORDS=60
  export MAX_SEQ_LEN="${max_seq:-196}"
  [[ "$MAX_SEQ_LEN" == "-" ]] && export MAX_SEQ_LEN=196
  export TWEET_MAX_SEQ_LEN=128
  [[ -n "$lr" && "$lr" != "-" ]] && export LR="$lr" || unset LR 2>/dev/null || true
  [[ -n "$ema" && "$ema" != "-" ]] && export EMA_DECAY="$ema" || unset EMA_DECAY 2>/dev/null || true
  [[ -n "$beta" && "$beta" != "-" ]] && export BETA_POLAR="$beta" || unset BETA_POLAR 2>/dev/null || true
  [[ -n "$ot" && "$ot" != "-" ]] && export OT_ALPHA="$ot" || unset OT_ALPHA 2>/dev/null || true
  [[ -n "$batch" && "$batch" != "-" ]] && export BATCH_SIZE="$batch" || unset BATCH_SIZE 2>/dev/null || true
  [[ -n "$epochs" && "$epochs" != "-" ]] && export EPOCHS="$epochs" || unset EPOCHS 2>/dev/null || true

  log "START $tag dust=$dust mass=$mass seeds=$seeds pat=$patience beta=${beta:--} ot=${ot:--} bs=${batch:--} ep=${epochs:--} ($note)"
  echo "- [$(date '+%H:%M:%S')] GPU$GPU_FILTER **$tag** seeds=$seeds ($note)" >>"$LOG_DIR/TASKS.md"

  local attempt rc=0
  for attempt in 1 2; do
    if bash "$ROOT/scripts/run_pgmner_vllm.sh" twitter2017; then
      rc=0
      break
    fi
    rc=$?
    log "FAIL $tag attempt=$attempt rc=$rc"
    [[ "$attempt" == 1 ]] && sleep 30 && export FORCE_RETRAIN=1 || true
  done
  export FORCE_RETRAIN=0

  python3 "$ROOT/scripts/collect_vllm_sweep.py" --write "$SUMMARY" 2>/dev/null || true
  if [[ "$rc" != "0" ]]; then
    log "GIVE UP $tag"
    echo "- [失败] GPU$GPU_FILTER **$tag** rc=$rc" >>"$LOG_DIR/TASKS.md"
    return "$rc"
  fi
  log "DONE $tag"
  echo "- [完成] GPU$GPU_FILTER **$tag**" >>"$LOG_DIR/TASKS.md"
  python3 "$ROOT/scripts/collect_vllm_sweep.py" --write "$LOG_DIR/BEST.md" --best-only 2>/dev/null || true
  return 0
}

log "VLLM queue phase=${PHASE:-default} RUNS_BASE=$RUNS_BASE"
[[ -f "$QUEUE" ]] || { log "missing $QUEUE"; exit 1; }

while IFS=$'\t' read -r gpu tag dust mass seeds patience desc_max max_seq lr ema beta ot batch epochs note; do
  gpu="${gpu//$'\r'/}"
  tag="${tag//$'\r'/}"
  [[ -z "${tag:-}" || "$tag" =~ ^# || "$tag" == "gpu" ]] && continue
  [[ "$gpu" != "$GPU_FILTER" ]] && continue
  run_job "$tag" "$dust" "$mass" "$seeds" "$patience" "$desc_max" "$max_seq" \
    "$lr" "$ema" "${beta:--}" "${ot:--}" "${batch:--}" "${epochs:--}" "${note:-}" || true
done <"$QUEUE"

log "queue worker finished"
python3 "$ROOT/scripts/collect_vllm_sweep.py" --write "$SUMMARY" 2>/dev/null || true
