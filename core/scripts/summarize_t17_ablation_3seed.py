#!/usr/bin/env python3
"""Aggregate t17 ablation results across seeds 123,456,789."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

KEYS = [
    "gw", "noppib", "nogota", "nonumstab", "nobalpolar",
    "fullot", "noema", "mass07", "beta0001", "beta005",
]
SEEDS = [123, 456, 789]
BASELINE_F1 = 0.9064
OUT_TSV = Path("outputs/t17_abl_3seed_summary.tsv")
OUT_MD = Path("outputs/t17_abl_3seed_summary.md")


def read_test_f1(out_dir: Path) -> float | None:
    hist = out_dir / "per_class_history.jsonl"
    if not hist.exists():
        return None
    rows = [json.loads(l) for l in hist.read_text().splitlines() if l.strip()]
    test = [r for r in rows if r.get("split") == "test"]
    if not test:
        return None
    return max(float(r["overall"]["f1"]) for r in test)


def main() -> None:
    lines = []
    md = [
        "# t17 Ablation — 3-seed summary (123, 456, 789)",
        "",
        f"Baseline gate: **{BASELINE_F1:.4f}** (dc10_m05 config)",
        "",
        "| key | seed123 | seed456 | seed789 | mean | std | max | vs baseline |",
        "|-----|---------|---------|---------|------|-----|-----|-------------|",
    ]
    for key in KEYS:
        f1s: dict[int, float | None] = {}
        for seed in SEEDS:
            out = Path(f"outputs/t17_abl_{key}_seed{seed}_v2")
            f1s[seed] = read_test_f1(out)
        vals = [v for v in f1s.values() if v is not None]
        mean = statistics.mean(vals) if vals else float("nan")
        std = statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else float("nan"))
        mx = max(vals) if vals else float("nan")
        vs = "PASS" if mx >= BASELINE_F1 else "below"
        row = {
            "key": key,
            "seed123": f1s.get(123),
            "seed456": f1s.get(456),
            "seed789": f1s.get(789),
            "mean": mean,
            "std": std,
            "max": mx,
            "vs_baseline": vs,
        }
        lines.append(row)
        fmt = lambda x: f"{x:.4f}" if x is not None else "—"
        md.append(
            f"| {key} | {fmt(f1s.get(123))} | {fmt(f1s.get(456))} | {fmt(f1s.get(789))} "
            f"| {mean:.4f} | {std:.4f} | {mx:.4f} | {vs} |"
        )

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    header = "key\tseed123\tseed456\tseed789\tmean\tstd\tmax\tvs_baseline\n"
    body = "".join(
        f"{r['key']}\t"
        + "\t".join(
            f"{r[f'seed{s}']:.4f}" if r[f"seed{s}"] is not None else "NA" for s in SEEDS
        )
        + f"\t{r['mean']:.4f}\t{r['std']:.4f}\t{r['max']:.4f}\t{r['vs_baseline']}\n"
        for r in lines
    )
    OUT_TSV.write_text(header + body)
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"Wrote {OUT_TSV} and {OUT_MD}")


if __name__ == "__main__":
    main()
