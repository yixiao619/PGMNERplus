#!/usr/bin/env bash
# A4 完成后串行：B1 → B3 → (下载 ViT-L) → B4 → [B2 if B1≥0.895] → summary
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/logs/phase1_5_serial.log"
mkdir -p "$ROOT/logs"
exec >>"$LOG" 2>&1

echo ""
echo "=== Phase 1.5 serial start $(date -Is) ==="

while pgrep -f 'main-mner-robert.py.*pp_ib_partial_ot_ema_focal_seed42' >/dev/null 2>&1; do
  ep=$(grep 'Training epoch num' outputs/pp_ib_partial_ot_ema_focal_seed42/train.log 2>/dev/null | tail -1 || true)
  echo "[wait A4] $(date -Is) $ep"
  sleep 120
done

while pgrep -f 'scripts/run_phase1_after_1024' >/dev/null 2>&1; do
  echo "[wait phase1 queue] $(date -Is)"
  sleep 30
done

_f1_from() {
  local out="$1"
  grep -v 'it/s\]' "$out/train.log" 2>/dev/null | grep 'best_score span f1' | tail -1 | awk '{print $NF}' || echo ''
}

echo "=== Task A4 done; best=$(_f1_from outputs/pp_ib_partial_ot_ema_focal_seed42) ==="

echo "=== Task B1 $(date -Is) ==="
bash scripts/run_pp_ib_full_ot_seed42.sh
B1_F1=$(_f1_from outputs/pp_ib_full_ot_seed42)
echo "B1 best F1=$B1_F1"

echo "=== Task B3 $(date -Is) ==="
bash scripts/run_pp_ib_lrtune_seed42.sh
B3_F1=$(_f1_from outputs/pp_ib_lrtune_seed42)
echo "B3 best F1=$B3_F1"

VIT_DIR="$ROOT/pretrain/clip-vit-large-patch14"
if [[ ! -f "$VIT_DIR/config.json" ]]; then
  echo "=== Download ViT-L/14 $(date -Is) ==="
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  if command -v hf >/dev/null 2>&1; then
    hf download openai/clip-vit-large-patch14 --local-dir "$VIT_DIR"
  else
    huggingface-cli download openai/clip-vit-large-patch14 --local-dir "$VIT_DIR"
  fi
fi

echo "=== Task B4 $(date -Is) ==="
bash scripts/run_pp_ib_vitL14_seed42.sh
B4_F1=$(_f1_from outputs/pp_ib_vitL14_seed42)
echo "B4 best F1=$B4_F1"

if awk -v f="$B1_F1" 'BEGIN{exit !(f+0 >= 0.895)}'; then
  echo "=== Task B2 (B1=$B1_F1 >= 0.895) $(date -Is) ==="
  if [[ -n "$B4_F1" ]] && awk -v f="$B4_F1" 'BEGIN{exit !(f+0 >= 0.895)}'; then
    export VIT_NAME=pretrain/clip-vit-large-patch14
    export BATCH_SIZE=6
    export EPOCH=40
  else
    export VIT_NAME=pretrain/clip-vit-base-patch16
    export BATCH_SIZE=16
    export EPOCH=40
  fi
  bash scripts/run_pp_ib_partial_ot_grid_seed42.sh
else
  echo "=== Skip B2 (B1=$B1_F1 < 0.895) ==="
fi

echo "=== Write summary $(date -Is) ==="
python3 scripts/write_phase1_5_summary.py

echo "=== Phase 1.5 serial done $(date -Is) ==="
