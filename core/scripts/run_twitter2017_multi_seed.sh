#!/usr/bin/env bash
# 多随机种子重复实验：baseline (CAC) / sem_ot / got
# 用法：cd MINER-My && bash scripts/run_twitter2017_multi_seed.sh
# （可加执行权限：chmod +x scripts/run_twitter2017_multi_seed.sh）
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEEDS=(13 42 2024)
MODES=(baseline sem_ot got)

COMMON=(
  --data_dir data/twitter2017
  --model_name_or_path pretrain/roberta-base
  --tokenizer_name pretrain/roberta-base
  --vit_name pretrain/clip-vit-base-patch16
  --gpu_id 0
  --epoch 30
  --batch_size 8
  --max_seq_len 128
  --max_span_len 4
  --lr 3e-5
  --bert_lr 3e-5
  --patience 0  --logging_steps 100
  --do_train true
  --do_eval true
  --do_predict true
  --ot_alpha 0.05
  --ot_eps 0.1
  --ot_sinkhorn_iter 10
  --ot_lambda_align 1.0
  --got_lambda_sem 0.5
  --got_struct_mode full
)

mkdir -p logs checkpoints

for MODE in "${MODES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "========== MODE=${MODE} SEED=${SEED} =========="
    case "${MODE}" in
      baseline)
        OUT="checkpoints/twitter2017-baseline-seed${SEED}"
        LOG="logs/run_twitter2017_baseline_seed${SEED}.log"
        EXTRA=()
        ;;
      sem_ot)
        OUT="checkpoints/twitter2017-sem_ot-seed${SEED}"
        LOG="logs/run_twitter2017_sem_ot_seed${SEED}.log"
        EXTRA=(--use_semantic_ot)
        ;;
      got)
        OUT="checkpoints/twitter2017-got-seed${SEED}"
        LOG="logs/run_twitter2017_got_seed${SEED}.log"
        EXTRA=(--use_got)
        ;;
      *)
        echo "Unknown MODE=${MODE}" >&2
        exit 1
        ;;
    esac
    mkdir -p "${OUT}"
    echo "  (训练详细日志: ${LOG})"
    set +e
    python main-mner-robert.py \
      "${COMMON[@]}" \
      --output_dir "${OUT}" \
      --seed "${SEED}" \
      "${EXTRA[@]}" \
      > "${LOG}" 2>&1
    EC=$?
    set -e
    if [ "${EC}" -ne 0 ]; then
      echo "FAILED MODE=${MODE} SEED=${SEED} (exit ${EC}), see ${LOG}" >&2
      exit "${EC}"
    fi
    echo "OK -> ${OUT} (${LOG})"
  done
done

echo "All runs finished. Aggregates: grep span_f1 logs/run_twitter2017_*_seed*.log ; dev/test metrics also under checkpoints/*/eval_results.txt etc."
