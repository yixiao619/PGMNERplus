#!/usr/bin/env bash
# 检查 CoNLL 文件与图片目录是否对齐
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"

DATASET="${1:-twitter2017}"
DATA_DIR="$MINER_MY_ROOT/data/$DATASET"
IMG_DIR="${DATA_DIR}_images"

python3 - "$DATA_DIR" "$IMG_DIR" <<'PY'
import sys
from pathlib import Path

data_dir, img_dir = Path(sys.argv[1]), Path(sys.argv[2])
ids = set()
for split in ("train", "dev", "test"):
    f = data_dir / f"{split}.txt"
    if not f.is_file():
        continue
    for line in f.read_text().splitlines():
        if line.startswith("IMGID:"):
            ids.add(line.split(":", 1)[1].strip())

missing = []
for img_id in sorted(ids):
    found = any((img_dir / f"{img_id}{ext}").is_file() for ext in (".jpg", ".jpeg", ".png"))
    if not found:
        missing.append(img_id)

print(f"tweets with IMGID: {len(ids)}")
print(f"missing images:    {len(missing)}")
if missing[:5]:
    print("examples:", ", ".join(missing[:5]))
PY
