#!/usr/bin/env bash
# A → C → B → D 顺序执行（上一段成功才进入下一段）；日志见各子目录 train.log 与 outputs/pgmner_chain.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

echo "======== PGMNER A→C→B→D chain start $(date -Is) ========"
bash scripts/run_dynamic_beta.sh
bash scripts/run_per_class_rerun_eval.sh
bash scripts/run_got_graph.sh
bash scripts/run_sensitivity_beta.sh
bash scripts/run_sensitivity_gw.sh
echo "======== PGMNER chain done $(date -Is) ========"
