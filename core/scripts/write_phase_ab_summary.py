#!/usr/bin/env python3
"""Scan Phase A/B output dirs and write outputs/phase_ab_summary.md."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    # Phase A
    ("t15_main_dc08_m06_seed42", "twitter2015", "outputs/t15_main_dc08_m06_seed42"),
    ("t15_main_dc08_m06_seed123", "twitter2015", "outputs/t15_main_dc08_m06_seed123"),
    ("t15_main_dc08_m06_seed456", "twitter2015", "outputs/t15_main_dc08_m06_seed456"),
    ("t15_probe_dc08_m05_seed42", "twitter2015", "outputs/t15_probe_dc08_m05_seed42"),
    ("t15_probe_dc08_m07_seed42", "twitter2015", "outputs/t15_probe_dc08_m07_seed42"),
    # Phase B — reuse existing full PGMNER
    ("pp_ib_partial_ot_grid_dc08_m06_seed42", "twitter2017", "outputs/pp_ib_partial_ot_grid_dc08_m06_seed42"),
    ("pp_ib_partial_ot_grid_dc08_m06_seed123", "twitter2017", "outputs/pp_ib_partial_ot_grid_dc08_m06_seed123"),
    ("pp_ib_partial_ot_grid_dc08_m06_seed456", "twitter2017", "outputs/pp_ib_partial_ot_grid_dc08_m06_seed456"),
    # Phase B ablations
    ("t17_abl_gw_seed42", "twitter2017", "outputs/t17_abl_gw_seed42"),
    ("t17_abl_gw_seed123", "twitter2017", "outputs/t17_abl_gw_seed123"),
    ("t17_abl_gw_seed456", "twitter2017", "outputs/t17_abl_gw_seed456"),
    ("t17_abl_noppib_seed42", "twitter2017", "outputs/t17_abl_noppib_seed42"),
    ("t17_abl_noppib_seed123", "twitter2017", "outputs/t17_abl_noppib_seed123"),
    ("t17_abl_noppib_seed456", "twitter2017", "outputs/t17_abl_noppib_seed456"),
    ("t17_abl_nogota_seed42", "twitter2017", "outputs/t17_abl_nogota_seed42"),
    ("t17_abl_nogota_seed123", "twitter2017", "outputs/t17_abl_nogota_seed123"),
    ("t17_abl_nogota_seed456", "twitter2017", "outputs/t17_abl_nogota_seed456"),
    ("t17_abl_nonumstab_seed42", "twitter2017", "outputs/t17_abl_nonumstab_seed42"),
    ("t17_abl_nobalpolar_seed42", "twitter2017", "outputs/t17_abl_nobalpolar_seed42"),
    ("t17_abl_fullot_seed42", "twitter2017", "outputs/t17_abl_fullot_seed42"),
    ("t17_abl_noema_seed42", "twitter2017", "outputs/t17_abl_noema_seed42"),
    ("t17_abl_nomiscfocal_seed42", "twitter2017", "outputs/t17_abl_nomiscfocal_seed42"),
]


def best_test_f1(out_dir: Path) -> str:
    er = out_dir / "eval_results.txt"
    if er.exists():
        text = er.read_text(errors="ignore")
        if "test best ckpt" in text:
            block = text.split("test best ckpt", 1)[-1]
            m = re.search(r"span_f1\s*=\s*([\d.]+)", block)
            if m:
                return m.group(1)
        vals = [float(x) for x in re.findall(r"span_f1\s*=\s*([\d.]+)", text)]
        if vals:
            return f"{max(vals):.5f}"
    tl = out_dir / "train.log"
    if tl.exists():
        text = tl.read_text(errors="ignore")
        if "test best ckpt" in text:
            block = text.split("test best ckpt", 1)[-1]
            m = re.search(r"span_f1\s*=\s*([\d.]+)", block)
            if m:
                return m.group(1)
        m = re.findall(r"best_score span f1[:\s]+([\d.]+)", text)
        if m:
            return m[-1]
    return "—"


def main() -> None:
    lines = [
        "# Phase A + Phase B summary",
        "",
        f"Generated: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}",
        "",
        "| run | dataset | best test F1 |",
        "|---|---|---:|",
    ]
    for run, dataset, rel in RUNS:
        f1 = best_test_f1(ROOT / rel)
        lines.append(f"| {run} | {dataset} | {f1} |")
    out = ROOT / "outputs" / "phase_ab_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
