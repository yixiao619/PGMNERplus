#!/usr/bin/env python3
"""汇总 PP-IB 5 seed 结果 → outputs/phase1_baseline_table.md"""
from __future__ import annotations

import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

RUNS = {
    "seed=42": "pp_ib_seed42",
    "seed=123": "pp_ib_seed123",
    "seed=456": "pp_ib_seed456",
    "seed=789": "pp_ib_seed789_v2",
    "seed=1024": "pp_ib_seed1024_v2",
}

FALLBACK_LOG = ROOT / "logs" / "phase2_pp_ib_v2.log"
MARKER_1024 = "PP-IB v2 seed=1024 only"


def _best_from_training_log(text: str) -> float | None:
    matches = re.findall(r"best_score\s+span\s+f1\s*[=:]\s*([0-9.]+)", text, flags=re.I)
    if matches:
        return float(matches[-1])

    er = re.search(
        r"\*\*\*\*\* Eval results test best ckpt \*\*\*\*\*[\s\S]*?"
        r"span_f1\s*=\s*([0-9.]+)",
        text,
    )
    if er:
        return float(er.group(1))

    best: float | None = None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "Save best model" not in line:
            continue
        for j in range(i - 1, max(i - 12, -1), -1):
            sm = re.search(r"span_f1\s*=\s*([0-9.]+)", lines[j])
            if sm:
                v = float(sm.group(1))
                best = v if best is None else max(best, v)
                break
    return best


def _log_text(dirname: str) -> str:
    log = OUTPUTS / dirname / "train.log"
    if log.exists():
        return log.read_text(encoding="utf-8", errors="ignore")
    if dirname == "pp_ib_seed1024_v2" and FALLBACK_LOG.exists():
        raw = FALLBACK_LOG.read_text(encoding="utf-8", errors="ignore")
        if MARKER_1024 in raw:
            raw = raw.split(MARKER_1024, 1)[-1]
        return raw
    er = OUTPUTS / dirname / "eval_results.txt"
    if er.exists():
        return er.read_text(encoding="utf-8", errors="ignore")
    return ""


def main() -> None:
    rows: list[tuple[str, str | float]] = []
    f1s: list[float] = []

    for name, dirname in RUNS.items():
        text = _log_text(dirname)
        if not text.strip():
            rows.append((name, "MISSING"))
            continue
        f1 = _best_from_training_log(text)
        if f1 is None:
            spans = [float(x) for x in re.findall(r"span_f1\s*=\s*([0-9.]+)", text)]
            if spans:
                f1 = max(spans)
                rows.append((name, f"{f1:.5f} (max eval, no final)"))
                f1s.append(f1)
                continue
            rows.append((name, "NO_BEST"))
            continue
        rows.append((name, f1))
        f1s.append(f1)

    md = [
        "# Phase 1 — PP-IB 5 seed baseline",
        "",
        "| Run | Best Test F1 |",
        "|---|---|",
    ]
    for name, f1 in rows:
        cell = f1 if isinstance(f1, str) else f"{f1:.5f}"
        md.append(f"| {name} | {cell} |")

    if f1s:
        md.append("")
        md.append(f"**Mean** = {statistics.mean(f1s):.5f}")
        if len(f1s) > 1:
            md.append(f"**Std**  = {statistics.stdev(f1s):.5f}")
        md.append(f"**Max**  = {max(f1s):.5f}")
        md.append(f"**n**    = {len(f1s)}")

    out = OUTPUTS / "phase1_baseline_table.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
