#!/usr/bin/env bash
# Fig.3 sweep — Step-A v2 配置，只扫 beta_polar 与 s_init（各 seed=42, 5 epoch）
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
GPU="${1:-2}"
OUT="$ROOT/results/fig3_sweep"
mkdir -p "$OUT/runs" logs

# Step-A v2 固定健康配置
COMMON=(
  --data_dir data/twitter2017
  --model_name_or_path pretrain/roberta-large
  --tokenizer_name pretrain/roberta-large
  --vit_name pretrain/clip-vit-base-patch16
  --use_hf_clip_vision true
  --auto_img_dims true
  --gpu_id "$GPU"
  --batch_size 16
  --max_seq_len 128
  --max_span_len 4
  --lr 3e-5
  --bert_lr 1e-5
  --warmup_ratio 0.1
  --logging_steps 999999
  --do_train true
  --do_eval false
  --do_predict false
  --patience 0
  --seed 42
  --epoch 5
  --use_pp_ib
  --pp_ib_gate_on_hidden
  --pp_ib_gate_init_std 0.1
  --pp_ib_gate_init_bias 1.0
  --pp_ib_lambda_balance 3
  --kl_weight 0.0002
  --delta_minus 0.2
  --delta_plus 0.8
  --beta_polar_warmup_epochs 0
  --use_got
  --got_mode partial_sinkhorn
  --ot_alpha 0.2
  --entropy_eps 0.1
  --dust_cost 0.5
  --ot_sinkhorn_iter 30
  --ot_warmup_epochs 3
)

extract_f1() {
  local log="$1"
  if grep -q 'best_score span f1:' "$log" 2>/dev/null; then
    grep 'best_score span f1:' "$log" | tail -1 | awk '{print $NF}'
    return
  fi
  grep -oE 'span_f1 = [0-9.]+' "$log" 2>/dev/null | tail -1 | awk '{print $NF}'
}

echo "beta_polar,f1" > "$OUT/beta.csv"
for B in 0.001 0.003 0.005 0.01 0.02 0.05; do
  RUN="$OUT/runs/beta_${B}"
  LOG="$OUT/beta_${B}.log"
  mkdir -p "$RUN"
  echo "=== beta_polar=$B ===" | tee "$LOG"
  python3 main-mner-robert.py \
    "${COMMON[@]}" \
    --output_dir "$RUN" \
    --beta_polar "$B" \
    --partial_mass_init 0.6 \
    2>&1 | tee -a "$LOG"
  F1="$(extract_f1 "$LOG")"
  F1="${F1:-nan}"
  echo "$B,$F1" >> "$OUT/beta.csv"
  echo "  -> F1=$F1"
done

echo "s_init,f1" > "$OUT/sinit.csv"
for S in 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  RUN="$OUT/runs/sinit_${S}"
  LOG="$OUT/sinit_${S}.log"
  mkdir -p "$RUN"
  echo "=== s_init=$S ===" | tee "$LOG"
  python3 main-mner-robert.py \
    "${COMMON[@]}" \
    --output_dir "$RUN" \
    --beta_polar 0.01 \
    --partial_mass_init "$S" \
    2>&1 | tee -a "$LOG"
  F1="$(extract_f1 "$LOG")"
  F1="${F1:-nan}"
  echo "$S,$F1" >> "$OUT/sinit.csv"
  echo "  -> F1=$F1"
done

echo "done -> $OUT/beta.csv , $OUT/sinit.csv"
python3 scripts/plot_fig3_sensitivity.py 2>&1 | tee -a logs/fig3_sweep_v2.log
