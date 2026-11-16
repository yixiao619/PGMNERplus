#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/jyx/yixiao/MINER-My
EXP="$ROOT/kais_extension/experiment20_overnight_ppib"
PY=/home/jyx/miniconda3/bin/python
PLOT_PY=/home/jyx/miniconda3/envs/nns-gpu/bin/python
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
GPU_UUID=GPU-2c45c36d-c789-4deb-7ac4-8712f14b82b6
GPU_BUS=00000000:B8:00.0
WALL_LIMIT=39600
GPU_LIMIT=28800
START_FILE="$EXP/results/scheduler_start_epoch.txt"
mkdir -p "$EXP/logs" "$EXP/results/runs" "$EXP/checkpoints" "$EXP/figures"
[[ -f "$START_FILE" ]] || date +%s > "$START_FILE"
START=$(cat "$START_FILE")
log(){ printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
elapsed(){ echo $(( $(date +%s)-START )); }
gpu_seconds(){ "$PY" - "$EXP/results/runs" <<'PY'
import json,sys
from pathlib import Path
s=0
for p in Path(sys.argv[1]).glob('*/status.json'):
 try:
  d=json.loads(p.read_text())
  if d.get('status')=='complete': s+=float(d.get('training_seconds',0))
 except: pass
print(int(s))
PY
}
within_budget(){ [[ $(elapsed) -lt $WALL_LIMIT && $(gpu_seconds) -lt $GPU_LIMIT ]]; }
verify_gpu(){
  local got
  got=$(nvidia-smi --id=2 --query-gpu=uuid,pci.bus_id --format=csv,noheader,nounits | tr -d ' ')
  [[ "$got" == "$GPU_UUID,$GPU_BUS" ]] || { log "GPU identity mismatch: $got"; exit 3; }
}
run_one(){
  local id=$1 extra=${2:-} status="$EXP/results/runs/$1/status.json"
  if [[ -f "$status" ]] && grep -q '"status": "complete"' "$status"; then log "SKIP complete $id"; return 0; fi
  within_budget || { log "BUDGET reached before $id"; return 2; }
  verify_gpu
  log "START $id on physical GPU2 UUID=$GPU_UUID (no idle wait requested)"
  CUDA_VISIBLE_DEVICES="$GPU_UUID" timeout --signal=TERM $((WALL_LIMIT-$(elapsed))) "$PY" -u "$EXP/scripts/run_one.py" --id "${id%_fixed}" $extra > "$EXP/logs/${id}.log" 2>&1
  local rc=$?; log "END $id rc=$rc"; "$PLOT_PY" "$EXP/scripts/summarize.py" || true; return $rc
}
log "Experiment 20 scheduler started; target physical GPU2 UUID=$GPU_UUID bus=$GPU_BUS"
verify_gpu
log "Running mandatory full-model smoke checks"
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PY" -u "$EXP/scripts/run_one.py" --id A0 --smoke > "$EXP/logs/smoke_A0.log" 2>&1 || { log "A0 smoke failed"; "$PLOT_PY" "$EXP/scripts/summarize.py"; exit 4; }
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PY" -u "$EXP/scripts/run_one.py" --id B1 --smoke > "$EXP/logs/smoke_B1.log" 2>&1 || { log "B1 smoke failed"; "$PLOT_PY" "$EXP/scripts/summarize.py"; exit 5; }
for id in A0 A1 A2 A3 A4 B1 B2; do
  within_budget || break
  run_one "$id" || true
  status="$EXP/results/runs/$id/status.json"
  if [[ -f "$status" ]] && "$PY" "$EXP/scripts/qualifies_control.py" "$status"; then
    run_one "${id}_fixed" "--fixed" || true
  else
    log "No matched fixed control triggered for $id"
  fi
done
"$PLOT_PY" "$EXP/scripts/summarize.py" || true
log "Experiment 20 scheduler finished wall_seconds=$(elapsed) gpu_training_seconds=$(gpu_seconds)"
