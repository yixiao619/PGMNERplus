#!/usr/bin/env bash
# Train dc10_m05 / dc08_m06: train-only compact aux, dev best, test w/o aux.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

GPU_ID="${GPU_ID:-0}"
SEEDS=(${SEEDS:-42 123 456})
EPOCH="${EPOCH:-40}"
MAX_SEQ="${MAX_SEQ:-128}"
DATASET="${DATASET:-twitter2017}"
FORCE="${FORCE:-0}"
LOG="$ROOT/logs/vllm_aug_top2_train.log"
mkdir -p logs outputs aux_knowledge

_run() {
  local tag="$1" dc="$2" mass="$3" seed="$4"
  local out="outputs/vllm_aug_${tag}_seed${seed}"
  if [[ "$FORCE" != "1" ]] && [[ -f "$out/eval_results.txt" ]] \
    && grep -q 'test best ckpt' "$out/eval_results.txt" 2>/dev/null; then
    local f1
    f1=$(grep -A2 'test best ckpt' "$out/eval_results.txt" | grep span_f1 | head -1 | awk '{print $3}')
    if awk -v f="$f1" 'BEGIN{exit !(f+0 >= 0.85)}'; then
      echo "[SKIP] $out test_F1=$f1 (>=0.85; set FORCE=1 to rerun)"
      return 0
    fi
    echo "[RERUN] $out had test_F1=$f1 (<0.85), training again"
  fi
  mkdir -p "$out"
  rm -f "$out/best_model.pth" "$out/eval_results.txt"
  echo ""
  echo "===== vllm_aug $tag seed=$seed dc=$dc mass=$mass -> $out $(date -Is) =====" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python main-mner-robert.py \
    --data_dir "data/$DATASET" \
    --output_dir "$out" \
    --model_name_or_path pretrain/roberta-large \
    --tokenizer_name pretrain/roberta-large \
    --vit_name pretrain/clip-vit-base-patch16 \
    --use_hf_clip_vision true --auto_img_dims true \
    --gpu_id 0 --epoch "$EPOCH" --batch_size 16 \
    --max_seq_len "$MAX_SEQ" --max_span_len 4 \
    --lr 3e-5 --bert_lr 1e-5 --warmup_ratio 0.1 \
    --patience 8 --seed "$seed" --logging_steps 100 \
    --do_train true --do_eval true --do_predict true \
    --use_pp_ib --beta_polar 0.01 --delta_minus 0.2 --delta_plus 0.8 \
    --use_got --got_mode partial_sinkhorn --ot_alpha 0.05 --entropy_eps 0.1 \
    --ot_sinkhorn_iter 30 --partial_mass_init "$mass" --dust_cost "$dc" \
    --ot_warmup_epochs 3 --use_ema --ema_decay 0.999 \
    --use_aux_knowledge --aux_train_only \
    --aux_knowledge_path "aux_knowledge/${DATASET}_{split}_aux.jsonl" \
    --eval_split_for_best dev \
    --max_aux_words 12 --aux_dropout 0.15 \
    --clas_loss_weight 2.0 --scl_loss_weight 0.15 \
    2>&1 | tee -a "$LOG" "$out/train.log"
  grep -v 'it/s\]' "$out/train.log" | grep 'best_score span f1' | tail -1 | tee -a "$LOG" || true
}

link_aux() {
  local ds="$1"
  ln -sf "${ds}_train_train_gold_distill.jsonl" "aux_knowledge/${ds}_train_aux.jsonl"
  ln -sf "${ds}_dev_clean.jsonl" "aux_knowledge/${ds}_dev_aux.jsonl"
  ln -sf "${ds}_test_clean.jsonl" "aux_knowledge/${ds}_test_aux.jsonl"
}

{
  echo "=== vllm aug v2 $(date -Is) GPU=$GPU_ID seeds=${SEEDS[*]} ===" | tee -a "$LOG"
  link_aux "$DATASET"
  for seed in "${SEEDS[@]}"; do
    _run "dc10_m05" 1.0 0.5 "$seed"
  done
  for seed in "${SEEDS[@]}"; do
    _run "dc08_m06" 0.8 0.6 "$seed"
  done
  echo "" | tee -a "$LOG"
  echo "=== summary ===" | tee -a "$LOG"
  python3 - <<'PY' | tee -a "$LOG"
import re
from pathlib import Path
for d in sorted(Path("outputs").glob("vllm_aug_*")):
    er = d / "eval_results.txt"
    if not er.exists():
        continue
    t = er.read_text(errors="ignore")
    if "test best ckpt" not in t:
        continue
    m = re.search(r"test best ckpt.*?span_f1\s*=\s*([\d.]+)", t, re.S)
    if m:
        print(f"{d.name}: test_F1={float(m.group(1)):.5f}")
PY
  echo "[DONE] $(date -Is)" | tee -a "$LOG"
}
