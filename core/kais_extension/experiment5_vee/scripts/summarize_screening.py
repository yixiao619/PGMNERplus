#!/usr/bin/env python3
"""Summarize Experiment 5 same-run screening results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "kais_extension" / "experiment5_vee"
RESULTS = EXP / "results"


RUNS = [
    ("PGMNER", "pgmner_baseline_seed42", 0),
    ("PGMNER+Mean", "vee_mean_seed42", None),
    ("PGMNER+Attention", "vee_attention_seed42", None),
    ("PGMNER+GatedAttention", "vee_gated_attention_seed42", None),
]


def load(name: str) -> dict | None:
    p = RESULTS / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_only", action="store_true")
    args = ap.parse_args()

    baseline = load("pgmner_baseline_seed42")
    baseline_f1 = float(baseline.get("test_span_f1", "nan")) if baseline else float("nan")
    rows = []
    for model, run_name, added in RUNS:
        d = load(run_name)
        if d is None:
            if args.baseline_only:
                continue
            rows.append(
                {
                    "model": model,
                    "seed": 42,
                    "best_dev_f1": "",
                    "test_precision": "",
                    "test_recall": "",
                    "test_f1": "",
                    "delta_f1_vs_same_run_baseline": "",
                    "best_epoch": "",
                    "parameter_count": "",
                    "added_parameters": "" if added is None else added,
                    "training_time": "",
                }
            )
            continue
        f1 = d.get("test_span_f1", "")
        delta = ""
        if f1 != "" and baseline and model != "PGMNER":
            delta = float(f1) - baseline_f1
        rows.append(
            {
                "model": model,
                "seed": 42,
                "best_dev_f1": d.get("dev_span_f1", ""),
                "test_precision": d.get("test_span_precision", ""),
                "test_recall": d.get("test_span_recall", ""),
                "test_f1": f1,
                "delta_f1_vs_same_run_baseline": delta,
                "best_epoch": d.get("best_epoch", ""),
                "parameter_count": d.get("parameter_count", ""),
                "added_parameters": d.get("added_parameters", "" if added is None else added),
                "training_time": d.get("training_time_sec", ""),
            }
        )

    out = RESULTS / "vee_screening_main.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "seed",
        "best_dev_f1",
        "test_precision",
        "test_recall",
        "test_f1",
        "delta_f1_vs_same_run_baseline",
        "best_epoch",
        "parameter_count",
        "added_parameters",
        "training_time",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

