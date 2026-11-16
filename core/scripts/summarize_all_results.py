#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 outputs/ 下各实验的 test span_f1（eval_results.txt best ckpt）。"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
TST = "***** Eval results test best ckpt *****"

BASELINE_NOTE = 0.906


def last_test_f1(eval_path: Path) -> Optional[float]:
    if not eval_path.is_file():
        return None
    text = eval_path.read_text(encoding="utf-8", errors="replace")
    parts = text.split(TST)
    if len(parts) < 2:
        return None
    m = re.search(r"span_f1 = ([\d.]+)", parts[-1])
    return float(m.group(1)) if m else None


def main():
    rows: list[dict] = []
    rows.append({"method": "Baseline (PP-IB ref)", "test_f1": "", "note": "beta=0.01 ~0.905–0.906"})

    gs = REPO_ROOT / "outputs/grid_search/results.csv"
    if gs.is_file():
        with open(gs, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                tf = r.get("test_f1") or ""
                if tf and tf != "NA":
                    rows.append(
                        {
                            "method": f"grid {r.get('phase')} beta={r.get('beta_polar')}",
                            "test_f1": tf,
                            "note": r.get("output_dir", ""),
                        }
                    )

    vitl = REPO_ROOT / "outputs/vitl_best_config/eval_results.txt"
    tf = last_test_f1(vitl)
    if tf is not None:
        rows.append({"method": "ViT-L best_config", "test_f1": str(tf), "note": str(vitl.parent)})

    ens = REPO_ROOT / "outputs/ensemble"
    if ens.is_dir():
        for sd in sorted(ens.glob("seed_*")):
            tf = last_test_f1(sd / "eval_results.txt")
            if tf is not None:
                rows.append({"method": f"ensemble {sd.name}", "test_f1": str(tf), "note": str(sd)})

    out_md = REPO_ROOT / "outputs/final_report.md"
    out_csv = REPO_ROOT / "outputs/all_results.csv"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "test_f1", "note"])
        w.writeheader()
        w.writerows(rows)

    numeric = []
    for r in rows:
        try:
            numeric.append((r["method"], float(r["test_f1"])))
        except (ValueError, TypeError):
            continue
    best = max(numeric, key=lambda x: x[1]) if numeric else None

    lines = ["# MNER F1 实验汇总\n", "\n", "| method | test_f1 | note |\n", "|---|---:|---|\n"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['test_f1']} | {r['note']} |\n")
    lines.append("\n")
    if best:
        lines.append(f"**Best in table**: {best[0]} → **{best[1]:.5f}**\n")
        lines.append(f"**vs reference {BASELINE_NOTE}**: {best[1] - BASELINE_NOTE:+.5f}\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
