#!/usr/bin/env bash
# 改进四: 多 seed。PGMNER改进 — seed=42,123,2024 各训一次，汇总 mean±stdev。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

BASE="${1:-outputs/pgmner_multiseed_$(date +%Y%m%d)}"
mkdir -p "$BASE"

SEEDS=(42 123 2024)
F1FILE="${BASE}/best_f1_values.txt"
: >"$F1FILE"

for s in "${SEEDS[@]}"; do
  OUT="${BASE}/seed_${s}"
  mkdir -p "$OUT"
  python main-mner-robert.py \
    --data_dir data/twitter2017 \
    --output_dir "$OUT" \
    --model_name_or_path pretrain/roberta-large \
    --tokenizer_name pretrain/roberta-large \
    --vit_name pretrain/clip-vit-base-patch16 \
    --gpu_id 0 \
    --epoch 30 \
    --batch_size 8 \
    --max_seq_len 128 \
    --max_span_len 4 \
    --patience 0 \
    --seed "$s" \
    --logging_steps 100 \
    --do_train true \
    --do_eval true \
    --do_predict true \
    --txt_global_dim 1024 \
    --txt_local_dim 1024 \
    --img_global_dim 768 \
    --img_local_dim 768 \
    --hidden_dim 1024 \
    --use_tiered_lr \
    --text_encoder_lr 2e-6 \
    --vision_encoder_lr 2e-6 \
    --main_lr 1e-5 \
    --warmup_ratio 0.1 \
    --use_pp_ib \
    --use_pp_ib_dpp \
    --beta_polar 0.005 \
    --beta_polar_warmup_epochs 3 \
    --use_got_cac \
    --lambda_gw 0.2 \
    --use_got_pp \
    --lambda_got_pp 0.05 \
    2>&1 | tee "$OUT/train.log"
  f=$(grep "best_score span f1" "$OUT/train.log" 2>/dev/null | tail -1 | sed -E 's/.*: *//' | tr -d ' \r' || true)
  echo "$f" >>"$F1FILE"
  echo "seed=$s best_f1_line tail: $f" | tee -a "$BASE/summary_seeds.txt"
done

python3 - <<PY
import statistics
from pathlib import Path
p = Path("$F1FILE")
lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
fs = []
for ln in lines:
    try:
        fs.append(float(ln))
    except ValueError:
        pass
if not fs:
    print("No numeric F1 parsed; see train.log in each seed_*", flush=True)
else:
    m = statistics.mean(fs)
    st = statistics.pstdev(fs) if len(fs) > 1 else 0.0
    agg = p.parent / "aggregate_f1.txt"
    agg.write_text(f"mean={m:.6f} stdev={st:.6f} values={fs}\\n")
    print(f"mean={m:.4f} stdev={st:.4f} n={len(fs)}  -> {agg}")
PY

echo "== multiseed done under $BASE =="
