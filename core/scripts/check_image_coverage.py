#!/usr/bin/env python3
"""
统计 data/<dataset>/{train,dev,test}.txt 中 IMGID 在对应图片目录下是否有 .jpg。
图片目录规则与训练一致：args.data_dir + "_images/"（相对仓库根）。

用法（在仓库根目录）：
  python scripts/check_image_coverage.py --data_dir data/twitter2015
  python scripts/check_image_coverage.py --data_dir data/twitter2017
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_imgids(txt_path: Path) -> list[str]:
    ids: list[str] = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("IMGID:"):
                ids.append(line.split(":", 1)[-1].strip())
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Check IMGID.jpg coverage under data_dir_images/")
    ap.add_argument(
        "--data_dir",
        type=str,
        default="data/twitter2017",
        help="e.g. data/twitter2017 or data/twitter2015",
    )
    ap.add_argument(
        "--repo_root",
        type=str,
        default="",
        help="Repository root (default: parent of scripts/)",
    )
    args = ap.parse_args()
    root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parent.parent
    data_dir = (root / args.data_dir).resolve()
    # 与 utils/datasets.py: img_path = args.data_dir + "_images/"
    name = data_dir.name
    img_root = (root / args.data_dir.rstrip("/")).parent / f"{name}_images"
    if not img_root.exists():
        img_root = root / f"{args.data_dir}_images"
    img_root = img_root.resolve()

    print(f"data_dir   : {data_dir}")
    print(f"images_dir : {img_root} (exists={img_root.exists()})")
    if img_root.is_symlink():
        print(f"  -> symlink target: {os.path.realpath(img_root)}")

    splits = ["train", "dev", "test"]
    for sp in splits:
        p = data_dir / f"{sp}.txt"
        if not p.is_file():
            print(f"\n[{sp}] skip (missing {p.name})")
            continue
        ids = parse_imgids(p)
        n = len(ids)
        miss = 0
        for i in ids:
            jpg = img_root / f"{i}.jpg"
            if not jpg.is_file():
                miss += 1
        hit = n - miss
        pct = 100.0 * hit / n if n else 0.0
        print(f"\n[{sp}] tweets={n}  jpg_found={hit}  missing={miss}  coverage={pct:.2f}%")
        if miss and n <= 5:
            for i in ids:
                ok = (img_root / f"{i}.jpg").is_file()
                print(f"  IMGID {i}: {'OK' if ok else 'MISSING'}")


if __name__ == "__main__":
    main()
