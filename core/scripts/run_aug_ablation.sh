#!/usr/bin/env bash
# Augmentation ablation matrix (serial). Default: smoke (1 seed, short epoch).
# Full: FULL=1 SEEDS="42 123 456" EPOCH=40 bash scripts/run_aug_ablation.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

GPU_ID="${GPU_ID:-0}"
SEEDS=(${SEEDS:-42})
EPOCH="${EPOCH:-5}"
DATASETS=(${DATASETS:-twitter2017})
MODES=(baseline clean verify)
LOG="$ROOT/logs/aug_ablation.log"
mkdir -p logs outputs

_run_one() {
  local ds="$1" mode="$2" ppib="$3" got="$4" seed="$5"
  local tag="aug_${ds}_${mode}_pp${ppib}_got${got}_s${seed}"
  local out="outputs/${tag}"
  local extra=()
  [[ "$ppib" == 1 ]] && extra+=(--use_pp_ib --beta_polar 0.01)
  [[ "$got" == 1 ]] && extra+=(--use_got --got_mode partial_sinkhorn --partial_mass_init 0.5 --dust_cost 1.0)
  if [[ "$mode" == clean ]]; then
    extra+=(--use_aux_knowledge --aux_knowledge_path "aux_knowledge/${ds}_{split}_clean.jsonl")
  fi
  if [[ -f "$out/eval_results.txt" ]] && grep -q 'test best ckpt' "$out/eval_results.txt" 2>/dev/null; then
    echo "[SKIP] $out"
    return 0
  fi
  echo "===== RUN $tag seed=$seed ====="
  CUDA_VISIBLE_DEVICES="$GPU_ID" python main-mner-robert.py \
    --data_dir "data/$ds" \
    --output_dir "$out" \
    --model_name_or_path pretrain/roberta-large \
    --tokenizer_name pretrain/roberta-large \
    --vit_name pretrain/clip-vit-base-patch16 \
    --use_hf_clip_vision true --auto_img_dims true \
    --gpu_id 0 --epoch "$EPOCH" --batch_size 16 \
    --seed "$seed" --patience 5 \
    --do_train true --do_eval true --do_predict true \
    "${extra[@]}" \
    2>&1 | tee "$out/train.log"
}

{
  echo "=== aug ablation $(date -Is) EPOCH=$EPOCH seeds=${SEEDS[*]} ==="
  for ds in "${DATASETS[@]}"; do
    for mode in "${MODES[@]}"; do
      for ppib in 0 1; do
        for got in 0 1; do
          for seed in "${SEEDS[@]}"; do
            _run_one "$ds" "$mode" "$ppib" "$got" "$seed" || true
          done
        done
      done
    done
  done
  echo "[DONE] aug ablation $(date -Is)"
} 2>&1 | tee -a "$LOG"
