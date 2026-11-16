#!/usr/bin/env python3
"""Extract logged transported-mass trajectories from historical logs."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "kais_extension" / "experiment2_mass_audit" / "results"
SCAN_ROOTS = [
    Path("/data/jyx/MINER-My/outputs"),
    ROOT,
    Path("/home/jyx/yixiao/MINER-My/artifacts"),
    Path("/home/jyx/yixiao/MINER-My/final_results"),
    Path("/home/jyx/yixiao/MINER-My/runs"),
]

PATTERN = re.compile(r"\[OT\]\s*epoch=(?P<epoch>\d+)\s+warmup=(?P<warmup>True|False)\s+partial_s=(?P<s>[0-9.]+)")


def s_to_logit(s: float) -> str:
    raw = (s - 0.3) / 0.6
    if raw <= 0.0 or raw >= 1.0:
        return ""
    return f"{math.log(raw / (1.0 - raw)):.12g}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".log", ".txt", ".out"}:
                continue
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line_no, line in enumerate(text.splitlines(), 1):
                m = PATTERN.search(line)
                if not m:
                    continue
                s = float(m.group("s"))
                epoch = m.group("epoch")
                warmup = m.group("warmup")
                rows.append({
                    "log_path": str(path),
                    "line": line_no,
                    "epoch": epoch,
                    "step": "",
                    "warmup": warmup,
                    "s_logit": "" if warmup == "True" else s_to_logit(s),
                    "s": f"{s:.12g}",
                    "raw_line": line.strip()[:500],
                })

    with (OUT_DIR / "historical_s_trajectories.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["log_path", "line", "epoch", "step", "warmup", "s_logit", "s", "raw_line"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    usable = [float(r["s"]) for r in rows if r["warmup"] != "True"]
    stats = {
        "status": "available" if rows else "Historical trajectory unavailable",
        "n_records": len(rows),
        "n_non_warmup_records": len(usable),
    }
    if usable:
        stats.update({
            "initial_s": usable[0],
            "minimum_s": min(usable),
            "maximum_s": max(usable),
            "final_s": usable[-1],
            "absolute_change": usable[-1] - usable[0],
            "relative_change": (usable[-1] - usable[0]) / usable[0],
            "mean_s": mean(usable),
        })
    (OUT_DIR / "historical_s_trajectory_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
