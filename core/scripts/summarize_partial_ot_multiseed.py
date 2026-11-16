#!/usr/bin/env python3
"""Summarize test F1 across all completed seeds for partial-OT grid tags."""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAGS = ("dc10_m05", "dc08_m06")
PARAMS = {
    "dc10_m05": (1.0, 0.5),
    "dc08_m06": (0.8, 0.6),
}


def best_f1(out_dir: Path) -> float | None:
    for name in ("train.log", "eval_results.txt"):
        p = out_dir / name
        if not p.exists():
            continue
        text = p.read_text(errors="ignore")
        if "test best ckpt" in text:
            m = re.search(r"test best ckpt.*?span_f1\s*=\s*([\d.]+)", text, re.S)
            if m:
                return float(m.group(1))
        ms = re.findall(r"best_score\s+span\s+f1\s*[=:]\s*([0-9.]+)", text, re.I)
        if ms:
            return float(ms[-1])
    return None


def collect_tag(tag: str) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    for d in sorted(ROOT.glob(f"outputs/pp_ib_partial_ot_grid_{tag}_seed*")):
        m = re.search(r"_seed(\d+)$", d.name)
        if not m:
            continue
        f1 = best_f1(d)
        if f1 is not None:
            rows.append((int(m.group(1)), f1))
    return sorted(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=list(DEFAULT_TAGS))
    args = ap.parse_args()

    print("| tag | dust | mass | n | mean | std | max | seeds (F1) |")
    print("|-----|-----:|-----:|--:|-----:|----:|----:|------------|")
    best_mean = None
    best_tag = None
    for tag in args.tags:
        rows = collect_tag(tag)
        if not rows:
            print(f"| {tag} | — | — | 0 | — | — | — | — |")
            continue
        vals = [f for _, f in rows]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        dc, mass = PARAMS.get(tag, (None, None))
        seed_str = ", ".join(f"{s}:{f:.5f}" for s, f in rows)
        print(
            f"| {tag} | {dc} | {mass} | {len(vals)} | {mean:.5f} | {std:.5f} | {max(vals):.5f} | {seed_str} |"
        )
        if best_mean is None or mean > best_mean:
            best_mean = mean
            best_tag = tag
    if best_tag:
        print(f"\nHighest mean among listed tags: {best_tag} ({best_mean:.5f})")


if __name__ == "__main__":
    main()
