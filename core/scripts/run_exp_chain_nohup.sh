#!/usr/bin/env bash
# 四组实验串行；后台：cd MINER-My && nohup bash scripts/run_exp_chain_nohup.sh &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs
LOG="${ROOT}/outputs/nohup_exp_chain.log"
SUMMARY="${ROOT}/outputs/exp_chain_summary.txt"
export PYTHONUNBUFFERED=1

{
  echo "======== chain start $(date -Is) ========"
  bash scripts/run_exp_baseline.sh outputs/exp1_baseline
  bash scripts/run_exp_pp_ib.sh outputs/exp2_pp_ib
  bash scripts/run_exp_got_cac.sh outputs/exp3_got_cac
  bash scripts/run_exp_both.sh outputs/exp4_both
  echo "======== 四组训练均结束 $(date -Is) ========"

  {
    echo "# PAR-MNER 四组对照汇总"
    echo "# 生成时间: $(date -Is)"
    echo ""
    for name in exp1_baseline exp2_pp_ib exp3_got_cac exp4_both; do
      d="outputs/$name"
      echo ""
      echo "##################################################################"
      echo "## $name  ($d)"
      echo "##################################################################"
      if [[ -f "$d/results.txt" ]]; then
        cat "$d/results.txt"
      else
        echo "(无 results.txt)"
      fi
      if [[ -f "$d/train.log" ]]; then
        echo ""
        echo "--- best_score（train.log 末几行）---"
        grep "best_score span f1" "$d/train.log" 2>/dev/null | tail -5 || true
      fi
      echo ""
    done
  } >"$SUMMARY"

  echo "======== 汇总已写入: $SUMMARY ========"
  cat "$SUMMARY"
  echo "======== chain done $(date -Is) ========"
} >>"$LOG" 2>&1
