#!/usr/bin/env bash
# 将 Twitter-2015/2017 文本与图片迁到 /data/jyx，并在 MINER-My/data 下保留同名软链。
# 支持 MINER-My/data/<name> 为真实目录或指向别处的符号链接（会先解析真实路径再 mv）。
# 若 data 下条目已被删但数据仍在 $MINER_DATASET_FALLBACK/<name>（默认 /data/jyx/datasets-miner），也会自动搬迁。
# 需要可写 /data/jyx（通常需在本机终端执行一次）：
#   sudo mkdir -p /data/jyx && sudo chown -R "$(id -un):$(id -gn)" /data/jyx
set -euo pipefail
DEST="${1:-/data/jyx}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/data"
FB="${MINER_DATASET_FALLBACK:-/data/jyx/datasets-miner}"

if [[ ! -w "$DEST" ]]; then
  echo "ERROR: 目录不可写: $DEST"
  echo "请先: sudo mkdir -p $DEST && sudo chown -R \$(id -un):\$(id -gn) $DEST"
  exit 1
fi

for d in twitter2015 twitter2015_images twitter2017 twitter2017_images; do
  target="$DEST/$d"
  src=""

  if [[ -L "$d" ]]; then
    src="$(readlink -f "$d" || true)"
    if [[ -z "$src" || ! -e "$src" ]]; then
      echo "ERROR: 断链或无效: $d"
      exit 1
    fi
    if [[ "$src" == "$target" ]]; then
      echo "OK已是目标: $d -> $target"
      continue
    fi
    rm -f "$d"
  elif [[ -d "$d" ]]; then
    src="$(realpath "$d")"
  elif [[ -d "$FB/$d" ]]; then
    src="$(realpath "$FB/$d")"
  fi

  if [[ -d "$target" ]]; then
    echo "目标已存在，跳过移动: $target"
  elif [[ -n "$src" && -d "$src" ]]; then
    mv "$src" "$target"
  else
    echo "ERROR: 缺少 $target（请保证 data/$d 存在、或为有效软链，或数据在 $FB/$d）"
    exit 1
  fi

  ln -sfn "$target" "$d"
  echo "symlink $d -> $target"
done

echo "完成。验证: python scripts/check_image_coverage.py --data_dir data/twitter2015"
