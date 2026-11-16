#!/usr/bin/env python3
"""Print LaTeX table row: PER LOC ORG MISC P R F1 for one or two datasets."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_eval(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    block = text.split("test best ckpt")[-1] if "test best ckpt" in text else text
    overall = {}
    for k in ("span_precision", "span_recall", "span_f1"):
        m = re.search(rf"{k} = ([\d.]+)", block)
        if m:
            overall[k] = float(m.group(1))
    per = {}
    for line in block.splitlines():
        m = re.match(
            r"\s+(PER|LOC|ORG|MISC):\s+P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)",
            line,
        )
        if m:
            per[m.group(1)] = float(m.group(4))
    return {"overall": overall, "per_class": per}


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}"


def row_for_dataset(run_dir: Path) -> str:
    ev = run_dir / "eval_results.txt"
    if not ev.is_file():
        return "—"
    d = parse_eval(ev)
    pc = d["per_class"]
    ov = d["overall"]
    cols = [fmt_pct(pc.get(t, 0.0)) for t in ("PER", "LOC", "ORG", "MISC")]
    cols += [
        fmt_pct(ov.get("span_precision", 0.0)),
        fmt_pct(ov.get("span_recall", 0.0)),
        fmt_pct(ov.get("span_f1", 0.0)),
    ]
    return " & ".join(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="runs/pgmner_vllm")
    ap.add_argument("--datasets", default="twitter2015,twitter2017")
    ap.add_argument("--seed", type=int, default=0, help="0 = best F1 across seeds")
    args = ap.parse_args()

    root = Path(args.runs_root)
    parts = []
    for ds in args.datasets.split(","):
        ds = ds.strip()
        ds_dir = root / ds
        if not ds_dir.is_dir():
            parts.append("—")
            continue
        if args.seed:
            parts.append(row_for_dataset(ds_dir / str(args.seed)))
        else:
            best = None
            best_f1 = -1.0
            for sd in sorted(ds_dir.iterdir()):
                if not sd.is_dir():
                    continue
                ev = sd / "eval_results.txt"
                if not ev.is_file():
                    continue
                f1 = parse_eval(ev)["overall"].get("span_f1", 0.0)
                if f1 > best_f1:
                    best_f1 = f1
                    best = sd
            parts.append(row_for_dataset(best) if best else "—")

    line = " & ".join(parts) + r" \\"
    print("% PGMNER+VLLM row (PER LOC ORG MISC P R F1 per dataset)")
    print(line)


if __name__ == "__main__":
    main()
