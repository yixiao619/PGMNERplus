#!/usr/bin/env bash
# 消融实验队列（Twitter-2017）
# 用法: GPU=0 bash scripts/train_ablation.sh
#       GPU=1 DATASET=t15 bash scripts/train_ablation.sh
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"

GPU="${GPU:-0}"
DATASET="${DATASET:-t17}"
MATRIX="$PGMNER_ROOT/config/ablation_matrix.tsv"

cp -f "$MATRIX" "$MINER_MY_ROOT/scripts/ablation_matrix.tsv"
cd "$MINER_MY_ROOT"
bash scripts/run_ablation.sh "$GPU" "$DATASET" scripts/ablation_matrix.tsv

echo ""
python3 "$PGMNER_ROOT/scripts/summarize_ablation.py" "$MINER_MY_ROOT"
