#!/usr/bin/env python3
"""扫描 outputs/ 下 Task A/B/C 各目录，提取 span F1，输出 outputs/phase2_summary.md"""
from __future__ import annotations

import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
RESULTS_FILE = OUTPUTS / "phase2_summary.md"


def _grep_best_f1(train_log: Path) -> float | None:
    if not train_log.exists():
        return None
    text = train_log.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(
        r"best_score\s+span\s+f1\s*[=:]\s*([0-9.]+)", text, flags=re.IGNORECASE
    )
    if not matches:
        return None
    return float(matches[-1])


def _grep_eval(out_dir: Path) -> dict:
    er = out_dir / "eval_results.txt"
    if not er.exists():
        return {}
    txt = er.read_text(encoding="utf-8", errors="ignore")
    res: dict = {}
    for tag in ("dev", "test"):
        pat = (
            rf"\*\*\*\*\* Eval results {tag} best ckpt \*\*\*\*\*[\s\S]*?"
            r"span_f1\s*=\s*([0-9.]+)[\s\S]*?"
            r"span_precision\s*=\s*([0-9.]+)[\s\S]*?"
            r"span_recall\s*=\s*([0-9.]+)"
        )
        m = re.search(pat, txt)
        if m:
            res[tag] = {
                "f1": float(m.group(1)),
                "p": float(m.group(2)),
                "r": float(m.group(3)),
            }
    return res


def scan(prefix: str) -> list[dict]:
    rows = []
    for d in sorted(OUTPUTS.glob(f"{prefix}*")):
        if not d.is_dir():
            continue
        f1 = _grep_best_f1(d / "train.log")
        ev = _grep_eval(d)
        rows.append(
            {
                "dir": d.name,
                "best_test_f1": f1,
                "dev_f1": ev.get("dev", {}).get("f1"),
                "test_f1": ev.get("test", {}).get("f1"),
            }
        )
    return rows


def _fmt(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.5f}"


def fmt_table(rows: list[dict], header: str) -> str:
    lines = [
        f"| {header} | dev F1 | test F1 (eval) | best_score (train.log) |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {dir:s} | {d} | {t} | {b} |".format(
                dir=r["dir"],
                d=_fmt(r["dev_f1"]),
                t=_fmt(r["test_f1"]),
                b=_fmt(r["best_test_f1"]),
            )
        )
    return "\n".join(lines)


def _seed_stats(rows: list[dict]) -> str:
    if not rows:
        return "_（尚未运行 Task A，无 `outputs/pp_ib_seed*`）_\n"
    vals = []
    for r in rows:
        v = r.get("test_f1") or r.get("best_test_f1")
        if v is not None:
            vals.append(float(v))
    if len(vals) < 2:
        return "_（有效 F1 不足 2 条，略过 mean/std）_\n"
    return (
        f"- **n** = {len(vals)}\n"
        f"- **max** = {max(vals):.5f}\n"
        f"- **mean** = {statistics.mean(vals):.5f}\n"
        f"- **std** = {statistics.pstdev(vals):.5f}\n"
    )


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    seed_rows = scan("pp_ib_seed")
    beta_rows = scan("pp_ib_beta")
    pot_rows = scan("pp_ib_pot") + scan("pp_ib_partial_ot")
    ref_rows = scan("baseline_fixed_b16") + scan("pp_ib_fixed_b16")

    out: list[str] = [
        "# Phase 2 实验汇总",
        "",
        "> 由 `scripts/summarize_phase2.py` 生成；跑完 Task A/B/C 后重新执行本脚本更新。",
        "",
        "## Task A · PP-IB 多 seed",
        "",
        _seed_stats(seed_rows),
        "",
        fmt_table(seed_rows, "run"),
        "",
        "## Task B · β_polar 网格",
        "",
        fmt_table(beta_rows, "β config"),
        "",
        "## Task C · PP-IB + Partial OT",
        "",
        fmt_table(pot_rows, "config"),
        "",
        "## 参考基线（task6）",
        "",
        fmt_table(ref_rows, "config"),
        "",
    ]

    RESULTS_FILE.write_text("\n".join(out), encoding="utf-8")
    print(f"[OK] Wrote {RESULTS_FILE}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
