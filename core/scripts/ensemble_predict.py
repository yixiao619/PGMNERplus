#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对多个目录下的 test_predictions.txt 做 token 级硬投票（预测列为第 3 列：word \\t gold \\t pred）。
输出与首模型相同格式的 ensemble 预测文件。
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import List, Tuple


def parse_predictions(path: Path) -> List[List[Tuple[str, str, str]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = text.strip().split("\n\n")
    sents = []
    for blk in blocks:
        rows = []
        for line in blk.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                continue
            rows.append((parts[0], parts[1], parts[2]))
        if rows:
            sents.append(rows)
    return sents


def vote_label(labels: list[str]) -> str:
    return Counter(labels).most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dirs",
        type=str,
        default="outputs/ensemble/seed_42,outputs/ensemble/seed_123,outputs/ensemble/seed_456,outputs/ensemble/seed_789,outputs/ensemble/seed_1024",
    )
    ap.add_argument("--output", type=str, default="outputs/ensemble/test_predictions_voted.txt")
    args = ap.parse_args()

    dirs = [Path(d.strip()) for d in args.model_dirs.split(",") if d.strip()]
    pred_files = [d / "test_predictions.txt" for d in dirs]
    for p in pred_files:
        if not p.is_file():
            raise SystemExit(f"Missing {p}")

    parsed = [parse_predictions(p) for p in pred_files]
    n = len(parsed[0])
    for i, pa in enumerate(parsed):
        if len(pa) != n:
            raise SystemExit(f"Sentence count mismatch: ref={n} vs dir {i}={len(pa)}")

    out_lines: list[str] = []
    for si in range(n):
        ref = parsed[0][si]
        m = len(ref)
        for pj in range(1, len(parsed)):
            if len(parsed[pj][si]) != m:
                raise SystemExit(f"Token count mismatch sent {si}")
        for ti in range(m):
            word, gold = ref[ti][0], ref[ti][1]
            preds = [parsed[pj][si][ti][2] for pj in range(len(parsed))]
            pred = vote_label(preds)
            out_lines.append(f"{word}\t{gold}\t{pred}")
        out_lines.append("")

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {outp} ({n} sentences, {len(dirs)} models)")


if __name__ == "__main__":
    main()
