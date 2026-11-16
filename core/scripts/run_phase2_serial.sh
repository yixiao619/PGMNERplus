#!/usr/bin/env bash
# Phase 2 串行：C1 → C2 → [C3 if C1 max≥0.901] → [C4 on best tag]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/logs/phase2_serial.log"
mkdir -p "$ROOT/logs"
exec >>"$LOG" 2>&1

_f1() {
  local d="$1"
  grep -v 'it/s\]' "$d/train.log" 2>/dev/null | grep 'best_score span f1' | tail -1 | awk '{print $NF}'
}

_c1_max() {
  local m=0 f
  for s in 42 123 456; do
    if [[ "$s" == "42" ]]; then
      D="$ROOT/outputs/pp_ib_partial_ot_grid_dc10_m06"
      [[ -f "$D/train.log" ]] || D="$ROOT/outputs/pp_ib_partial_ot_grid_dc10_m06_seed42"
    else
      D="$ROOT/outputs/pp_ib_partial_ot_grid_dc10_m06_seed${s}"
    fi
    f=$(_f1 "$D")
    if [[ -n "$f" ]] && awk -v a="$m" -v b="$f" 'BEGIN{exit !(b>a)}'; then
      m=$f
    fi
  done
  echo "$m"
}

echo ""
echo "=== Phase 2 serial start $(date -Is) ==="

echo "=== C1 seed=123 $(date -Is) ==="
bash scripts/run_b2_winning_seed123.sh

echo "=== C1 seed=456 $(date -Is) ==="
bash scripts/run_b2_winning_seed456.sh

C1_MAX=$(_c1_max)
echo "C1 max-of-3=$C1_MAX"

echo "=== C2 fine grid $(date -Is) ==="
for tag in dc10_m05 dc10_m055 dc08_m06 dc12_m06 dc07_m065; do
  echo "--- C2 $tag ---"
  bash "scripts/run_b2_${tag}_seed42.sh"
done

C2_BEST=0
C2_BEST_TAG=""
for tag in dc10_m06 dc10_m05 dc10_m055 dc08_m06 dc12_m06 dc07_m065; do
  if [[ "$tag" == "dc10_m06" ]]; then
    D="$ROOT/outputs/pp_ib_partial_ot_grid_dc10_m06"
  else
    D="$ROOT/outputs/pp_ib_partial_ot_grid_${tag}_seed42"
  fi
  f=$(_f1 "$D")
  echo "C2 check $tag F1=$f"
  if [[ -n "$f" ]] && awk -v a="$C2_BEST" -v b="$f" 'BEGIN{exit !(b>a)}'; then
    C2_BEST=$f
    C2_BEST_TAG=$tag
  fi
done
echo "C2 best=$C2_BEST tag=$C2_BEST_TAG"

# ----- C2 完整 grid 汇总(供论文热图) -----
echo "=== C2 grid dump $(date -Is) ==="
{
  echo "# Partial OT grid (B2 + C2)"
  echo ""
  echo "| tag | test F1 | dir |"
  echo "|---|---:|---|"
  for tag in dc10_m08 dc05_m08 dc10_m06 dc20_m07 dc10_m05 dc10_m055 dc08_m06 dc12_m06 dc07_m065 dc10_m045; do
    D=""
    for cand in "$ROOT/outputs/pp_ib_partial_ot_grid_${tag}" \
                "$ROOT/outputs/pp_ib_partial_ot_grid_${tag}_seed42"; do
      [[ -d "$cand" && -f "$cand/train.log" ]] && D="$cand" && break
    done
    if [[ -z "$D" ]]; then
      echo "| ${tag} | — | MISSING |"
      continue
    fi
    F1=$(_f1 "$D")
    echo "| ${tag} | ${F1:-—} | \`$(basename "$D")\` |"
  done
} | tee "$ROOT/outputs/partial_ot_grid_all.md"

# ----- 旧 C3/C4(dc10_m06 基底)作废; 新峰 dc10_m05 → 见 scripts/run_phase2_d_series.sh -----
# if awk -v m="$C1_MAX" 'BEGIN{exit !(m+0 >= 0.901)}'; then
#   bash scripts/run_b2_winning_vitL14_seed42.sh
# fi
# bash scripts/run_b2_winning_ema9995_seed42.sh
# python3 scripts/write_phase2_summary.py

echo "=== Phase 2 C2 queue done $(date -Is); auto-start D-series ==="
PHASE2_CHAINED=1 bash scripts/run_phase2_d_series.sh

echo "=== Phase 2 full queue done (C2 + D-series) $(date -Is) ==="
