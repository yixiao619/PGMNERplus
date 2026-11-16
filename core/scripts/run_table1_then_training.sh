#!/usr/bin/env bash
# 顺序：① Table1 全量数据分析 ② 生成图 ③ towards_91（默认仅 EXP-A；ViT-L 设 RUN_EXP_BCDE=true）
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p analysis/logs analysis/results
export PYTHONUNBUFFERED=1

echo "[1/3] Table1 dataset analysis (Twitter-2015/2017) ..."
python analysis/dataset_analysis.py \
  --data_root data \
  --output_dir analysis/results \
  --clip_model pretrain/clip-vit-base-patch16 \
  --splits train,dev,test \
  --datasets Twitter-2015,Twitter-2017 \
  2>&1 | tee analysis/logs/dataset_analysis_full.log

echo "[2/3] Figures ..."
python analysis/make_table1_figures.py 2>&1 | tee -a analysis/logs/dataset_analysis_full.log

echo "[3/3] towards_91 (RUN_EXP_BCDE=${RUN_EXP_BCDE:-false}) ..."
RUN_EXP_BCDE="${RUN_EXP_BCDE:-false}" bash scripts/run_towards_91.sh 2>&1 | tee analysis/logs/towards_91_after_analysis.log

echo "Done $(date -Is). Summaries: analysis/results/table1_summary.json outputs/towards_91/"
