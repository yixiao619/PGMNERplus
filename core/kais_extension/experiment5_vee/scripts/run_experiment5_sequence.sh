#!/usr/bin/env bash
# Sequential Experiment 5 launcher. It waits for one idle GPU and then runs
# baseline followed by VEE variants only if the user explicitly starts it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1

GPU_ID="$(bash kais_extension/experiment5_vee/scripts/wait_for_idle_gpu.sh)"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

python - <<'PY'
import torch
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
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

