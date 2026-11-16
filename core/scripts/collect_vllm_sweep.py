#!/usr/bin/env python3
"""Collect dev/test span F1 from runs/pgmner_vllm/* (primary metric: dev F1)."""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def runs_roots() -> list[Path]:
    roots = [
        Path("/data/jyx/MINER-My/runs/pgmner_vllm"),
        ROOT / "runs" / "pgmner_vllm",
    ]
    seen: set[str] = set()
    out: list[Path] = []
    for p in roots:
        key = str(p.resolve())
        if key not in seen and p.is_dir():
            seen.add(key)
            out.append(p)
    return out


def parse_split_f1(eval_path: Path, split_label: str) -> dict[str, float] | None:
    if not eval_path.is_file():
        return None
    text = eval_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        rf"Eval results {re.escape(split_label)} best ckpt \*+\s*"
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


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for runs_root in runs_roots():
        _collect_from_root(rows, runs_root)
    return rows


def _collect_from_root(rows: list[dict], runs_root: Path) -> None:
    for exp_dir in sorted(runs_root.iterdir()):
        if not exp_dir.is_dir() or "novllm" in exp_dir.name:
            continue
        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            eval_path = seed_dir / "eval_results.txt"
            dev = parse_split_f1(eval_path, "dev")
            test = parse_split_f1(eval_path, "test")
            if not dev and not test:
                continue
            report = (seed_dir / "report.txt").read_text(errors="replace") if (seed_dir / "report.txt").is_file() else ""
            use_vllm = (
                "use_vllm_desc=1" in report
                or exp_dir.name == "twitter2017"
                or exp_dir.name.startswith("twitter2017_")
            )
            try:
                rel = str(seed_dir.relative_to(ROOT))
            except ValueError:
                rel = str(seed_dir)
            rows.append(
                {
                    "experiment": exp_dir.name,
                    "seed": seed_dir.name,
                    "use_vllm": use_vllm,
                    "dev_f1": dev["f1"] if dev else None,
                    "dev_p": dev["p"] if dev else None,
                    "dev_r": dev["r"] if dev else None,
                    "test_f1": test["f1"] if test else None,
                    "test_p": test["p"] if test else None,
                    "test_r": test["r"] if test else None,
                    "path": rel,
                }
            )


def _pct(x: float | None) -> str:
    return f"{x * 100:.2f}%" if x is not None else "—"


def sort_key(r: dict) -> tuple:
    # Rank by dev F1 first; fall back to test F1
    dev = r["dev_f1"] if r["dev_f1"] is not None else -1.0
    test = r["test_f1"] if r["test_f1"] is not None else -1.0
    return (-dev, -test, r["experiment"], r["seed"])


def format_table(rows: list[dict]) -> str:
    lines = [
        f"# PGMNER VLLM sweep ({datetime.now().isoformat(timespec='seconds')})",
        "",
        "Primary metric: **dev span F1** (same as checkpoint selection when `eval_split_for_best=dev`).",
        "",
        "| experiment | seed | dev F1 | test F1 | dev P/R | path |",
        "|------------|-----:|-------:|--------:|---------|------|",
    ]
    for r in sorted(rows, key=sort_key):
        pr = f"{_pct(r['dev_p'])}/{_pct(r['dev_r'])}" if r["dev_f1"] is not None else "—"
        lines.append(
            f"| {r['experiment']} | {r['seed']} | {_pct(r['dev_f1'])} | {_pct(r['test_f1'])} | {pr} | `{r['path']}` |"
        )
    best: dict[str, dict] = {}
    for r in rows:
        if r["experiment"] not in best or sort_key(r) < sort_key(best[r["experiment"]]):
            best[r["experiment"]] = r
    if best:
        lines += ["", "## Best seed per experiment (by dev F1)", ""]
        for name in sorted(best):
            b = best[name]
            lines.append(
                f"- **{name}**: seed {b['seed']} dev={_pct(b['dev_f1'])} test={_pct(b['test_f1'])}"
            )
    return "\n".join(lines) + "\n"


def format_best(rows: list[dict]) -> str:
    vllm = [r for r in rows if r["use_vllm"] and r["dev_f1"] is not None]
    if not vllm:
        vllm = [r for r in rows if r["use_vllm"] and r["test_f1"] is not None]
        metric = "test"
        key = "test_f1"
    else:
        metric = "dev"
        key = "dev_f1"
    if not vllm:
        return "# No finished VLLM runs yet\n"
    best = max(vllm, key=lambda x: x[key] or -1.0)
    lines = [
        f"# Best VLLM run ({datetime.now().isoformat(timespec='seconds')})",
        "",
        f"Ranking: **{metric} span F1**",
        "",
        f"**{best['experiment']}** seed **{best['seed']}**",
        f"- dev F1: **{_pct(best['dev_f1'])}** (P={_pct(best['dev_p'])} R={_pct(best['dev_r'])})",
        f"- test F1: **{_pct(best['test_f1'])}**",
        "",
        f"Path: `{best['path']}`",
        "",
        "## Top 5 (by dev F1)",
        "",
    ]
    for r in sorted(vllm, key=sort_key)[:5]:
        lines.append(
            f"- {r['experiment']} s{r['seed']}: dev **{_pct(r['dev_f1'])}** / test {_pct(r['test_f1'])}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", type=Path, help="write full summary markdown")
    ap.add_argument("--append", type=Path, help="append refresh")
    ap.add_argument("--best-only", action="store_true", help="write only best VLLM row")
    args = ap.parse_args()
    rows = collect_rows()
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        text = format_best(rows) if args.best_only else format_table(rows)
        args.write.write_text(text)
        print(f"Wrote {args.write}")
    if args.append:
        args.append.parent.mkdir(parents=True, exist_ok=True)
        with args.append.open("a") as f:
            f.write(f"\n<!-- refresh {datetime.now().isoformat(timespec='seconds')} -->\n")
            f.write(format_table(rows))


if __name__ == "__main__":
    main()
