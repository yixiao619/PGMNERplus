#!/usr/bin/env bash
# 将 UMT README 中的 Google Drive 图片包解压并把 *.jpg 收集到 data/twitter2015_images/
# 用法（在仓库根目录）：
#   bash scripts/setup_twitter2015_images_from_zip.sh /path/to/downloaded.zip
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ZIP="${1:?用法: bash scripts/setup_twitter2015_images_from_zip.sh /path/to/UMT_images.zip}"

TARGET="$ROOT/data/twitter2015_images"
mkdir -p "$TARGET"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

unzip -q -o "$ZIP" -d "$TMP"
# 收集任意子目录下 jpg（UMT 包结构可能多一层）
find "$TMP" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) -print0 | while IFS= read -r -d '' f; do
  base=$(basename "$f")
  cp -f "$f" "$TARGET/$base"
done

n=$(find "$TARGET" -maxdepth 1 -type f -iname '*.jpg' | wc -l)
echo "已复制到 $TARGET 的 .jpg 数量: $n"
echo "运行覆盖率检查："
python scripts/check_image_coverage.py --data_dir data/twitter2015
