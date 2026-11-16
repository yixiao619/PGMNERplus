#!/usr/bin/env bash
# Force-start Experiment 5 on a specified visible physical GPU.
# This does not kill or modify any other GPU process.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

GPU_ID="${1:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

echo "===== Experiment 5 no-wait launch $(date -Is) ====="
echo "physical_gpu=${GPU_ID}"
command -v nvidia-smi || true
nvidia-smi || true

python - <<'PY'
import torch
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("logical cuda:0", torch.cuda.get_device_name(0))
PY

run_one() {
  local name="$1"
  local vee="$2"
  local log="kais_extension/experiment5_vee/logs/${name}.log"
  echo "===== RUN ${name} vee=${vee} physical_gpu=${GPU_ID} $(date -Is) =====" | tee "$log"
  python kais_extension/experiment5_vee/scripts/run_experiment5_model.py \
    --run_name "$name" \
    --vee_type "$vee" \
    2>&1 | tee -a "$log"
}

run_one pgmner_baseline_seed42 none

python kais_extension/experiment5_vee/scripts/summarize_screening.py --baseline_only

python - <<'PY'
import json
from pathlib import Path
p = Path("kais_extension/experiment5_vee/results/pgmner_baseline_seed42.json")
d = json.loads(p.read_text())
f1 = float(d.get("test_span_f1", -1))
print("same-run baseline test F1", f1)
if f1 < 0.895:
    raise SystemExit("Baseline below ACCEPTABLE_REPRODUCTION; stop before VEE.")
PY

run_one vee_mean_seed42 mean
run_one vee_attention_seed42 attention
run_one vee_gated_attention_seed42 gated_attention

python kais_extension/experiment5_vee/scripts/summarize_screening.py
