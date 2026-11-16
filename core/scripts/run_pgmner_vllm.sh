#!/usr/bin/env bash
# PGMNER + VLLM: generate descriptions -> train (3 seeds) -> eval -> LaTeX row
# Usage:
#   VLLM_BACKEND=stub bash scripts/run_pgmner_vllm.sh twitter2017
#   VLLM_BACKEND=openai OPENAI_API_KEY=... bash scripts/run_pgmner_vllm.sh twitter2017
# Repro baseline (no desc): USE_VLLM_DESC=0 bash scripts/run_pgmner_vllm.sh twitter2017
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

DATASET="${1:-twitter2017}"
GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-42,123,456}"
VLLM_BACKEND="${VLLM_BACKEND:-qwen35}"
QWEN35_PATH="${QWEN35_PATH:-/home/LargeModels/Qwen/Qwen3.5-27B}"
# 无 OpenAI key 时不要静默用 stub/structured 冒充 VLLM
if [[ "$VLLM_BACKEND" == "openai" ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
  source "$(dirname "$0")/load_openai_key.sh" 2>/dev/null || true
fi
if [[ "$VLLM_BACKEND" == "openai" ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[WARN] 无 OPENAI_API_KEY，改用本地 Qwen3.5: $QWEN35_PATH"
  VLLM_BACKEND=qwen35
fi
if [[ "$VLLM_BACKEND" == "stub" ]]; then
  echo "[WARN] backend=stub 仅为占位，不是真实 VLLM；请用 VLLM_BACKEND=qwen35"
fi
USE_VLLM_DESC="${USE_VLLM_DESC:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
RUN_TAG="${RUN_TAG:-}"
DUST_COST="${DUST_COST:-1.0}"
PARTIAL_MASS="${PARTIAL_MASS:-0.5}"
PATIENCE="${PATIENCE:-8}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-3e-5}"
BERT_LR="${BERT_LR:-1e-5}"
EMA_DECAY="${EMA_DECAY:-0.999}"
DESC_MAX_WORDS="${DESC_MAX_WORDS:-60}"
TWEET_MAX_SEQ="${TWEET_MAX_SEQ_LEN:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-196}"
BETA_POLAR="${BETA_POLAR:-0.01}"
OT_ALPHA="${OT_ALPHA:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RUNS_ROOT="${RUNS_BASE:-$ROOT/runs/pgmner_vllm}"

DATA_DIR="data/${DATASET}"
if [[ -n "$RUN_TAG" ]]; then
  RUNS="${RUNS_ROOT}/${DATASET}_${RUN_TAG}"
else
  RUNS="${RUNS_ROOT}/${DATASET}"
fi
mkdir -p logs "$RUNS" "${DATA_DIR}/vllm_desc"

echo "=== PGMNER+VLLM dataset=$DATASET tag=${RUN_TAG:-default} backend=$VLLM_BACKEND use_desc=$USE_VLLM_DESC dust=$DUST_COST patience=$PATIENCE ==="

if [[ "${SKIP_GEN:-0}" == "1" ]]; then
  echo "----- [1/3] Skip generation (SKIP_GEN=1) -----"
elif [[ "$USE_VLLM_DESC" == "1" ]]; then
  echo "----- [1/3] Generate VLLM descriptions -----"
  for split in train dev test; do
    GEN_EXTRA=()
    [[ "${FRESH_DESC:-0}" == "1" ]] && GEN_EXTRA+=(--fresh)
    python3 scripts/generate_vllm_desc.py \
      --dataset "$DATASET" \
      --split "$split" \
      --backend "$VLLM_BACKEND" \
      --model_path "${QWEN35_PATH:-/home/LargeModels/Qwen/Qwen3.5-27B}" \
      --data_dir "$DATA_DIR" \
      --qps 0 \
      "${GEN_EXTRA[@]}" \
      2>&1 | tee -a "logs/vllm_desc_${DATASET}_${split}.log"
  done
fi

VLLM_ARGS=()
if [[ "$USE_VLLM_DESC" == "1" ]]; then
  VLLM_ARGS+=(
    --use_vllm_desc
    --vllm_desc_path "${DATA_DIR}/vllm_desc/{split}.jsonl"
    --desc_max_words "$DESC_MAX_WORDS"
    --tweet_max_seq_len "$TWEET_MAX_SEQ"
    --max_seq_len "$MAX_SEQ_LEN"
  )
else
  VLLM_ARGS+=(--max_seq_len 128 --tweet_max_seq_len 128)
fi

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
echo "----- [2/3] Train seeds: ${SEED_ARR[*]} -----"
for seed in "${SEED_ARR[@]}"; do
  OUT="${RUNS}/${seed}"
  mkdir -p "$OUT"
  if [[ "${FORCE_RETRAIN:-0}" != "1" ]] && grep -q 'Eval results test best ckpt' "$OUT/eval_results.txt" 2>/dev/null; then
    echo "[SKIP] seed=$seed already done"
    continue
  fi
  echo "===== train seed=$seed -> $OUT ====="
  python main-mner-robert.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT" \
    --model_name_or_path pretrain/roberta-large \
    --tokenizer_name pretrain/roberta-large \
    --vit_name pretrain/clip-vit-base-patch16 \
    --use_hf_clip_vision true --auto_img_dims true \
    --gpu_id 0 \
    --epoch "$EPOCHS" --batch_size "$BATCH_SIZE" \
    --max_span_len 4 \
    --lr "$LR" --bert_lr "$BERT_LR" \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --patience "$PATIENCE" --seed "$seed" \
    --logging_steps 100 \
    --eval_split_for_best dev \
    --do_train true --do_eval true --do_predict true \
    --use_pp_ib --beta_polar "$BETA_POLAR" \
    --delta_minus 0.2 --delta_plus 0.8 \
    --use_got --got_mode partial_sinkhorn \
    --ot_alpha "$OT_ALPHA" --entropy_eps 0.1 \
    --ot_sinkhorn_iter 30 \
    --partial_mass_init "$PARTIAL_MASS" --dust_cost "$DUST_COST" \
    --ot_warmup_epochs 3 \
    --use_ema --ema_decay "$EMA_DECAY" \
    "${VLLM_ARGS[@]}" \
    2>&1 | tee "$OUT/train.log"

  {
    echo "seed=$seed dataset=$DATASET run_tag=${RUN_TAG:-default}"
    echo "use_vllm_desc=$USE_VLLM_DESC dust_cost=$DUST_COST patience=$PATIENCE desc_max=$DESC_MAX_WORDS"
    echo "vllm_backend=$VLLM_BACKEND"
    if [[ -f "${DATA_DIR}/vllm_desc/generation_cost.jsonl" ]]; then
      echo "--- generation cost ---"
      tail -5 "${DATA_DIR}/vllm_desc/generation_cost.jsonl"
    fi
    if [[ -f "$OUT/eval_results.txt" ]]; then
      grep -A20 "test best ckpt" "$OUT/eval_results.txt" | head -25
    fi
  } > "$OUT/report.txt"
done

echo "----- [3/3] LaTeX row (best seed per dataset) -----"
python3 scripts/print_pgmner_vllm_latex.py --runs_root runs/pgmner_vllm --datasets "$DATASET"
echo "Done. Runs under $RUNS"
