#!/usr/bin/env python3
"""从 eval_results.txt 读取 dev/test span 指标。"""
from __future__ import annotations

import argparse
import re
import sys


def read_block(text: str, label: str) -> dict[str, float] | None:
    m = re.search(
        rf"Eval results {re.escape(label)} best ckpt \*+\s*"
        r"span_f1\s*=\s*([\d.]+)\s*"
        r"span_precision\s*=\s*([\d.]+)\s*"
        r"span_recall\s*=\s*([\d.]+)",
        text,
        re.MULTILINE,
    )
    if not m:
        return None
    f1, p, r = (float(m.group(i)) for i in (1, 2, 3))
    return {"f1": f1, "p": p, "r": r}


def read_per_class(text: str, label: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    parts = text.split("*****")
    for part in parts:
        if f"{label} best ckpt" not in part:
            continue
        in_pc = False
        for line in part.splitlines():
            if line.strip() == "per_class_f1:":
                in_pc = True
                continue
            if not in_pc or ":" not in line:
                continue
            m = re.match(
                r"\s*(\w+):\s+P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)",
                line,
            )
            if m:
                out[m.group(1)] = {
                    "p": float(m.group(2)),
                    "r": float(m.group(3)),
                    "f1": float(m.group(4)),
                }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_file")
    ap.add_argument("--per-class", action="store_true")
    args = ap.parse_args()

    text = open(args.eval_file, encoding="utf-8", errors="replace").read()
    for split in ("dev", "test"):
        m = read_block(text, split)
        if not m:
            continue
        print(
            f"{split:4s}  P={m['p']*100:.2f}%  R={m['r']*100:.2f}%  "
            f"F1={m['f1']*100:.2f}%  ({m['p']:.5f} {m['r']:.5f} {m['f1']:.5f})"
        )
        if args.per_class:
            pc = read_per_class(text, split)
            for lab in sorted(pc):
                x = pc[lab]
                print(f"  {lab:4s}  F1={x['f1']*100:.2f}%")

    if not read_block(text, "test"):
        print("未找到 test best ckpt", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
