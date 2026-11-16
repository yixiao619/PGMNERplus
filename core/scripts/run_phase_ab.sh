#!/usr/bin/env bash
# Phase A (Twitter-2015) + Phase B ablations (Twitter-2017), serial single-GPU queue.
# Launch: nohup bash scripts/run_phase_ab.sh > outputs/phase_ab.console.log 2>&1 &
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
  python main-mner-robert.py \
    --model_name_or_path pretrain/roberta-large \
    --tokenizer_name pretrain/roberta-large \
    --vit_name pretrain/clip-vit-base-patch16 \
    --use_hf_clip_vision true --auto_img_dims true \
    --gpu_id 0 --epoch 40 --batch_size 16 --max_seq_len 128 --max_span_len 4 \
    --lr 3e-5 --bert_lr 1e-5 --warmup_ratio 0.1 --patience 0 --logging_steps 100 \
    --do_train true --do_eval true --do_predict true \
    "$@" \
    2>&1 | tee "$out/train.log"
}

# Reference template deltas (Twitter-2017 main): PP-IB + partial_sinkhorn + EMA
_REF_PP_IB=(
  --use_pp_ib --beta_polar 0.01 --delta_minus 0.2 --delta_plus 0.8
)
_REF_GOT=(
  --use_got --got_mode partial_sinkhorn --ot_alpha 0.05 --entropy_eps 0.1
  --ot_sinkhorn_iter 30 --partial_mass_init 0.6 --dust_cost 0.8 --ot_warmup_epochs 3
)
_REF_EMA=(--use_ema --ema_decay 0.999)

echo "=== Phase A+B queue start $(date -Is) ==="

# ---------- Phase A: Twitter-2015 ----------
for spec in \
  "t15_main_dc08_m06_seed42:42:0.6" \
  "t15_main_dc08_m06_seed123:123:0.6" \
  "t15_main_dc08_m06_seed456:456:0.6" \
  "t15_probe_dc08_m05_seed42:42:0.5" \
  "t15_probe_dc08_m07_seed42:42:0.7"; do
  IFS=':' read -r tag seed mass <<< "$spec"
  _run_one "$tag" "outputs/${tag}" \
    --data_dir data/twitter2015 \
    --seed "$seed" \
    "${_REF_PP_IB[@]}" \
    "${_REF_GOT[@]}" \
    --partial_mass_init "$mass" \
    "${_REF_EMA[@]}" \
    --output_dir "outputs/${tag}"
done

# ---------- Phase B: Twitter-2017 ablations ----------
for seed in 42 123 456; do
  _run_one "t17_abl_gw_seed${seed}" "outputs/t17_abl_gw_seed${seed}" \
    --data_dir data/twitter2017 \
    --seed "$seed" \
    "${_REF_PP_IB[@]}" \
    --use_got --got_mode gw --ot_alpha 0.05 --entropy_eps 0.1 --ot_sinkhorn_iter 30 \
    --partial_mass_init 0.6 --dust_cost 0.8 --ot_warmup_epochs 3 \
    "${_REF_EMA[@]}" \
    --output_dir "outputs/t17_abl_gw_seed${seed}"
done

for seed in 42 123 456; do
  _run_one "t17_abl_noppib_seed${seed}" "outputs/t17_abl_noppib_seed${seed}" \
    --data_dir data/twitter2017 \
    --seed "$seed" \
    "${_REF_GOT[@]}" \
    "${_REF_EMA[@]}" \
    --output_dir "outputs/t17_abl_noppib_seed${seed}"
done

for seed in 42 123 456; do
  _run_one "t17_abl_nogota_seed${seed}" "outputs/t17_abl_nogota_seed${seed}" \
    --data_dir data/twitter2017 \
    --seed "$seed" \
    "${_REF_PP_IB[@]}" \
    "${_REF_EMA[@]}" \
    --output_dir "outputs/t17_abl_nogota_seed${seed}"
done

_run_one "t17_abl_nonumstab_seed42" "outputs/t17_abl_nonumstab_seed42" \
  --data_dir data/twitter2017 --seed 42 \
  "${_REF_PP_IB[@]}" \
  --beta_polar_warmup_epochs 0 --pp_ib_gate_init_std 0 --pp_ib_logvar_clamp 0 \
  "${_REF_GOT[@]}" \
  "${_REF_EMA[@]}" \
  --output_dir outputs/t17_abl_nonumstab_seed42

_run_one "t17_abl_nobalpolar_seed42" "outputs/t17_abl_nobalpolar_seed42" \
  --data_dir data/twitter2017 --seed 42 \
  "${_REF_PP_IB[@]}" --pp_ib_lambda_balance 0 \
  "${_REF_GOT[@]}" \
  "${_REF_EMA[@]}" \
  --output_dir outputs/t17_abl_nobalpolar_seed42

_run_one "t17_abl_fullot_seed42" "outputs/t17_abl_fullot_seed42" \
  --data_dir data/twitter2017 --seed 42 \
  "${_REF_PP_IB[@]}" \
  --use_got --got_mode node_sinkhorn --ot_alpha 0.05 --entropy_eps 0.1 --ot_sinkhorn_iter 30 \
  --partial_mass_init 0.6 --dust_cost 0.8 --ot_warmup_epochs 3 \
  "${_REF_EMA[@]}" \
  --output_dir outputs/t17_abl_fullot_seed42

_run_one "t17_abl_noema_seed42" "outputs/t17_abl_noema_seed42" \
  --data_dir data/twitter2017 --seed 42 \
  "${_REF_PP_IB[@]}" \
  "${_REF_GOT[@]}" \
  --output_dir outputs/t17_abl_noema_seed42

_run_one "t17_abl_nomiscfocal_seed42" "outputs/t17_abl_nomiscfocal_seed42" \
  --data_dir data/twitter2017 --seed 42 \
  "${_REF_PP_IB[@]}" \
  "${_REF_GOT[@]}" \
  "${_REF_EMA[@]}" \
  --output_dir outputs/t17_abl_nomiscfocal_seed42

echo ""
echo "=== Phase A+B runs finished $(date -Is); writing summary ==="
python3 scripts/write_phase_ab_summary.py
echo "=== Phase A+B queue done $(date -Is) ==="
