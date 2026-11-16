#!/usr/bin/env bash
# Live progress for aux generation / ablation logs.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GEN_LOG="${1:-$ROOT/logs/aug_generate_twitter2017.log}"
ABL_LOG="${2:-$ROOT/logs/aug_ablation.log}"

watch -n 10 "
echo '=== $(date -Is) aux pipeline ==='
echo '--- generation ---'
if [[ -f '$GEN_LOG' ]]; then
  grep -E 'progress|finished|split=|DONE|ERROR' '$GEN_LOG' | tail -8
  for f in $ROOT/aux_knowledge/twitter2017_*_clean.jsonl; do
    [[ -f \"\$f\" ]] && echo \"\$(basename \$f): \$(wc -l <\"\$f\") lines\"
  done
else
  echo '(no gen log)'
fi
echo '--- ablation ---'
if [[ -f '$ABL_LOG' ]]; then
  grep -E 'RUN |SKIP|best_score|DONE' '$ABL_LOG' | tail -6
else
  echo '(no ablation log)'
fi
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -3
"
