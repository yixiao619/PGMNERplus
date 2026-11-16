#!/usr/bin/env bash
# Wait for a genuinely idle GPU without disturbing existing jobs.
set -euo pipefail

INTERVAL_SEC="${INTERVAL_SEC:-300}"
CONFIRM_SEC="${CONFIRM_SEC:-60}"
MIN_FREE_MB="${MIN_FREE_MB:-102400}"
MAX_UTIL="${MAX_UTIL:-10}"
GPU_LIST="${GPU_LIST:-0 1 2}"
LOG="${LOG:-kais_extension/experiment5_vee/logs/gpu_wait.log}"

mkdir -p "$(dirname "$LOG")"

snapshot() {
  local tag="$1"
  {
    echo "===== ${tag} $(date -Is) ====="
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
    echo "--- processes ---"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true
  } | tee -a "$LOG" >&2
}

gpu_has_process() {
  local gpu_index="$1"
  local uuid
  uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F, -v g="$gpu_index" '$1 ~ g {gsub(/^ +| +$/, "", $2); print $2}')"
  if [[ -z "$uuid" ]]; then
    return 1
  fi
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, -v u="$uuid" -v min_mem=1024 '
        {for (i=1;i<=NF;i++) gsub(/^ +| +$/, "", $i)}
        $1 == u && ($3 + 0) >= min_mem {found=1}
        END {exit found ? 0 : 1}
      '
}

gpu_meets_thresholds() {
  local gpu_index="$1"
  local line free util
  line="$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F, -v g="$gpu_index" '{idx=$1; gsub(/^ +| +$/, "", idx); if (idx == g) print $0}')"
  [[ -n "$line" ]] || return 1
  free="$(echo "$line" | awk -F, '{gsub(/^ +| +$/, "", $2); print $2}')"
  util="$(echo "$line" | awk -F, '{gsub(/^ +| +$/, "", $3); print $3}')"
  [[ "$free" -ge "$MIN_FREE_MB" && "$util" -le "$MAX_UTIL" ]] || return 1
  if gpu_has_process "$gpu_index"; then
    return 1
  fi
  return 0
}

while true; do
  snapshot "check"
  for gpu in $GPU_LIST; do
    if gpu_meets_thresholds "$gpu"; then
      echo "candidate_gpu=${gpu}; confirming after ${CONFIRM_SEC}s" | tee -a "$LOG" >&2
      sleep "$CONFIRM_SEC"
      snapshot "confirm"
      if gpu_meets_thresholds "$gpu"; then
        echo "GPU_AVAILABLE=YES physical_gpu=${gpu}" | tee -a "$LOG" >&2
        echo "$gpu"
        exit 0
      fi
      echo "candidate_gpu=${gpu} failed confirmation" | tee -a "$LOG" >&2
    fi
  done
  echo "GPU_AVAILABLE=NO; sleeping ${INTERVAL_SEC}s" | tee -a "$LOG" >&2
  sleep "$INTERVAL_SEC"
done
