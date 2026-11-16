#!/usr/bin/env bash
# Fig.3 hyperparameter sensitivity probes on GPU2 (Twitter-2017, seed=42), serial.
# Launch: nohup bash scripts/run_fig3_gpu2.sh > outputs/fig3_gpu2.console.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

_run_done() {
  local d="$1"
  if [[ -f "$d/eval_results.txt" ]] && grep -q 'Eval results test best ckpt' "$d/eval_results.txt" 2>/dev/null; then
    return 0
  fi
  if [[ -f "$d/train.log" ]] && grep -q 'Eval results test best ckpt' "$d/train.log" 2>/dev/null; then
    return 0
  fi
  return 1
}

_run_one() {
  local name="$1"
  local out="$2"
  shift 2
  mkdir -p "$out"
  if _run_done "$out"; then
    echo "[SKIP] $name ($out) already has test best ckpt"
    return 0
  fi
  echo ""
  echo "===== RUN $name -> $out $(date -Is) ====="
  CUDA_VISIBLE_DEVICES=2 nice -n 10 python main-mner-robert.py \
    --data_dir data/twitter2017 \
    --model_name_or_path pretrain/roberta-large \
    --tokenizer_name pretrain/roberta-large \
    --vit_name pretrain/clip-vit-base-patch16 \
    --use_hf_clip_vision true --auto_img_dims true \
    --gpu_id 0 --epoch 40 --batch_size 16 --max_seq_len 128 --max_span_len 4 \
    --lr 3e-5 --bert_lr 1e-5 --warmup_ratio 0.1 --patience 0 --seed 42 --logging_steps 100 \
    --do_train true --do_eval true --do_predict true \
    --use_pp_ib --beta_polar 0.01 --delta_minus 0.2 --delta_plus 0.8 \
    --use_got --got_mode partial_sinkhorn --ot_alpha 0.05 --entropy_eps 0.1 --ot_sinkhorn_iter 30 \
    --partial_mass_init 0.6 --dust_cost 0.8 --ot_warmup_epochs 3 \
    --use_ema --ema_decay 0.999 \
    "$@" \
    2>&1 | tee "$out/train.log"
}

echo "=== Fig.3 GPU2 queue start $(date -Is) ==="

# A. beta_polar sweep (dust=0.8, mass=0.6 fixed)
for beta in 0.001 0.003 0.005 0.02 0.05; do
  tag="t17_fig3_beta${beta}_seed42"
  _run_one "$tag" "outputs/${tag}" \
    --beta_polar "$beta" \
    --output_dir "outputs/${tag}"
done

# B. partial_mass_init sweep (dust=0.8, beta_polar=0.01 fixed)
for mass in 0.3 0.4 0.5 0.7 0.8 0.9; do
  tag="t17_fig3_mass${mass}_dc08_seed42"
  _run_one "$tag" "outputs/${tag}" \
    --partial_mass_init "$mass" \
    --output_dir "outputs/${tag}"
done

echo ""
echo "=== Fig.3 GPU2 runs finished $(date -Is); printing plot dicts ==="
python3 <<'PY'
import re
from pathlib import Path

ROOT = Path(".")

def best_test_f1(out_dir: Path) -> float | None:
    er = out_dir / "eval_results.txt"
    if er.exists():
        text = er.read_text(errors="ignore")
        if "test best ckpt" in text:
            m = re.search(r"span_f1\s*=\s*([\d.]+)", text.split("test best ckpt", 1)[-1])
            if m:
                return float(m.group(1))
    tl = out_dir / "train.log"
    if tl.exists() and "test best ckpt" in tl.read_text(errors="ignore"):
        m = re.search(
            r"test best ckpt.*?span_f1\s*=\s*([\d.]+)",
            tl.read_text(errors="ignore"),
            re.S,
        )
        if m:
            return float(m.group(1))
    return None

ANCHOR = ROOT / "outputs/pp_ib_partial_ot_grid_dc08_m06_seed42"
anchor_f1 = best_test_f1(ANCHOR)

beta_vals = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05]
beta_f1 = {}
for b in beta_vals:
    if b == 0.01:
        beta_f1[b] = anchor_f1
        continue
    tag = f"t17_fig3_beta{b}_seed42"
    beta_f1[b] = best_test_f1(ROOT / "outputs" / tag)

mass_vals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
sinit_f1 = {}
for m in mass_vals:
    if m == 0.6:
        sinit_f1[m] = anchor_f1
        continue
    tag = f"t17_fig3_mass{m}_dc08_seed42"
    sinit_f1[m] = best_test_f1(ROOT / "outputs" / tag)

def fmt(d):
    parts = []
    for k in sorted(d.keys()):
        v = d[k]
        parts.append(f"{k!r}: {v!r}" if v is None else f"{k}: {v}")
    return "{" + ", ".join(parts) + "}"

print("")
print("# paste into plot_fig3_sensitivity.py")
print(f"BETA_F1={fmt(beta_f1)}")
print(f"SINIT_F1={fmt(sinit_f1)}")
print("")
print("| sweep | value | test F1 |")
print("|---|---|---:|")
for b in beta_vals:
    print(f"| beta_polar | {b} | {beta_f1[b] or '—'} |")
for m in mass_vals:
    print(f"| partial_mass_init | {m} | {sinit_f1[m] or '—'} |")
PY

echo "=== Fig.3 GPU2 queue done $(date -Is) ==="
