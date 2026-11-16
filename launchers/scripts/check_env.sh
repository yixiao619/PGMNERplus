#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"

DATASET="${1:-twitter2017}"
DATA_DIR="$MINER_MY_ROOT/data/$DATASET"
IMG_DIR="${DATA_DIR}_images"

echo "MINER-My: $MINER_MY_ROOT"
echo "data:     $DATA_DIR"
echo "images:   $IMG_DIR"

ok=0
die() { echo "[x] $1"; ok=1; }
pass() { echo "[ok] $1"; }

[[ -f "$MINER_MY_ROOT/main-mner-robert.py" ]] && pass main-mner-robert.py || die main-mner-robert.py
for split in train dev test; do
  [[ -f "$DATA_DIR/${split}.txt" ]] && pass "${split}.txt" || die "$DATA_DIR/${split}.txt"
done
[[ -d "$IMG_DIR" ]] && pass images || die "$IMG_DIR"
for p in pretrain/roberta-large pretrain/clip-vit-base-patch16; do
  [[ -d "$MINER_MY_ROOT/$p" ]] && pass "$p" || die "$MINER_MY_ROOT/$p"
done

python3 - <<'PY' || die "python deps"
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY

exit "$ok"
