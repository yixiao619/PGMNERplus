#!/usr/bin/env python3
"""汇总 outputs/t17_abl_* 各 seed 的 test F1。"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

KEYS = [
    "gw", "noppib", "nogota", "nonumstab", "nobalpolar",
    "fullot", "noema", "mass07", "beta0001", "beta005",
]
SEEDS = [123, 456, 789]


def test_f1(run_dir: Path) -> float | None:
    hist = run_dir / "per_class_history.jsonl"
    if hist.is_file():
        rows = [json.loads(l) for l in hist.read_text().splitlines() if l.strip()]
        test = [r for r in rows if r.get("split") == "test"]
        if test:
            return max(float(r["overall"]["f1"]) for r in test)
    ev = run_dir / "eval_results.txt"
    if not ev.is_file():
        return None
    import re
    m = re.search(r"Eval results test best ckpt.*?span_f1\s*=\s*([\d.]+)", ev.read_text(), re.S)
    return float(m.group(1)) if m else None


def main() -> None:
    miner = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../MINER-My")
    out_root = miner / "outputs"
    print(f"# ablation summary  ({out_root})")
    print("| key | 123 | 456 | 789 | mean | max |")
    print("|-----|-----|-----|-----|------|-----|")
    for key in KEYS:
        f1s = {}
        for s in SEEDS:
            d = out_root / f"t17_abl_{key}_seed{s}_v2"
            f1s[s] = test_f1(d)
        vals = [v for v in f1s.values() if v is not None]
        mean = statistics.mean(vals) if vals else float("nan")
        mx = max(vals) if vals else float("nan")
        fmt = lambda x: f"{x:.4f}" if x is not None else "-"
        print(
            f"| {key} | {fmt(f1s.get(123))} | {fmt(f1s.get(456))} | {fmt(f1s.get(789))} "
            f"| {mean:.4f} | {mx:.4f} |"
        )


if __name__ == "__main__":
    main()
