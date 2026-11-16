#!/usr/bin/env bash
# Real-time PGMNER+VLLM progress (generation + training)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="${1:-twitter2017}"
DATA_DIR="$ROOT/data/${DATASET}"
RUNS="$ROOT/runs/pgmner_vllm/${DATASET}"

expect_train=3373
expect_dev=723
expect_test=723
if [[ "$DATASET" == "twitter2015" ]]; then
  expect_train=4000
  expect_dev=1000
  expect_test=3257
fi

echo "=== $(date -Is) PGMNER+VLLM monitor ($DATASET) ==="
echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
echo ""
echo "--- Processes ---"
pgrep -af "generate_vllm_desc|run_pgmner_vllm|main-mner-robert" 2>/dev/null | grep -v monitor || echo "(none)"
echo ""
echo "--- VLLM description cache (lines / expected) ---"
for split in train dev test; do
  f="$DATA_DIR/vllm_desc/${split}.jsonl"
  if [[ -f "$f" ]]; then
    n=$(wc -l < "$f")
    exp_var="expect_${split}"
    exp=${!exp_var}
    pct=$(awk -v n="$n" -v e="$exp" 'BEGIN{printf "%.1f", (e>0?100*n/e:0)}')
    echo "  $split: $n / $exp (${pct}%)"
  else
    echo "  $split: missing"
  fi
done
if [[ -f "$DATA_DIR/vllm_desc/generation_cost.jsonl" ]]; then
  echo "  last cost log:"
  tail -2 "$DATA_DIR/vllm_desc/generation_cost.jsonl" | sed 's/^/    /'
fi
echo ""
echo "--- Generation log (tail) ---"
tail -5 "$ROOT/logs/vllm_desc_${DATASET}_openai.log" 2>/dev/null | sed 's/^/  /' || \
  tail -5 "$ROOT/logs/vllm_desc_${DATASET}_all.log" 2>/dev/null | sed 's/^/  /' || echo "  (no log)"
echo ""
echo "--- Pipeline log (tail) ---"
tail -8 "$ROOT/logs/pgmner_vllm_openai_pipeline.log" 2>/dev/null | grep -vE 'it/s|\[A' | sed 's/^/  /' || echo "  (no pipeline log)"
echo ""
echo "--- Training runs ---"
if [[ -d "$RUNS" ]]; then
  for sd in "$RUNS"/*; do
    [[ -d "$sd" ]] || continue
    seed=$(basename "$sd")
    if grep -q "test best ckpt" "$sd/eval_results.txt" 2>/dev/null; then
      tf=$(grep -A1 "test best ckpt" "$sd/eval_results.txt" | grep "span_f1" | awk '{print $3}')
      echo "  seed $seed: DONE test_F1=$tf"
    elif [[ -f "$sd/train.log" ]]; then
      bf=$(grep "best_score span f1" "$sd/train.log" 2>/dev/null | tail -1 | awk '{print $NF}')
      ep=$(grep -c "epoch_num:" "$sd/train.log" 2>/dev/null || echo "?")
      echo "  seed $seed: training dev_best≈$bf (epochs logged: $ep)"
    else
      echo "  seed $seed: pending"
    fi
  done
else
  echo "  (no runs yet)"
fi
