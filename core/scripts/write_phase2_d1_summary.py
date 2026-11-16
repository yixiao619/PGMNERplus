#!/usr/bin/env python3
"""outputs/phase2_d1_summary.md — C2 grid + D1/D1b multiseed + decision table."""
from __future__ import annotations

import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "phase2_d1_summary.md"

GRID_TAGS = [
    "dc10_m08",
    "dc05_m08",
    "dc10_m06",
    "dc20_m07",
    "dc10_m05",
    "dc10_m045",
    "dc10_m055",
    "dc08_m06",
    "dc12_m06",
    "dc07_m065",
]
PARAMS = {
    "dc10_m08": (1.0, 0.8),
    "dc05_m08": (0.5, 0.8),
    "dc10_m06": (1.0, 0.6),
    "dc20_m07": (2.0, 0.7),
    "dc10_m05": (1.0, 0.5),
    "dc10_m045": (1.0, 0.45),
    "dc10_m055": (1.0, 0.55),
    "dc08_m06": (0.8, 0.6),
    "dc12_m06": (1.2, 0.6),
    "dc07_m065": (0.7, 0.65),
}


def best_f1(d: Path) -> float | None:
    log = d / "train.log"
    if not log.exists():
        return None
    m = re.findall(
        r"best_score\s+span\s+f1\s*[=:]\s*([0-9.]+)", log.read_text(errors="ignore"), re.I
    )
    return float(m[-1]) if m else None


def find_grid_dir(tag: str) -> Path | None:
    for name in (f"pp_ib_partial_ot_grid_{tag}", f"pp_ib_partial_ot_grid_{tag}_seed42"):
        p = ROOT / "outputs" / name
        if (p / "train.log").exists():
            return p
    return None


def multiseed_stats(prefix: str) -> tuple[dict[int, float | None], float | None, float | None, float | None]:
    f1s: dict[int, float | None] = {}
    for s in (42, 123, 456):
        p = ROOT / "outputs" / f"pp_ib_partial_ot_grid_{prefix}_seed{s}"
        f1s[s] = best_f1(p)
    vals = [v for v in f1s.values() if v is not None]
    if not vals:
        return f1s, None, None, None
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f1s, mean, std, max(vals)


def decision_row(m05: float | None, m08: float | None) -> str:
    a = m05 if m05 is not None else -1.0
    b = m08 if m08 is not None else -1.0
    hi = 0.905
    lo = 0.900
    if a >= hi and b >= hi:
        main = "dc10_m05" if a >= b else "dc08_m06"
        return (
            f"两者均 ≥0.905 → **主推 {main}** (max 更高), 另一格作 ablation/sensitivity; 可进 **D2 ViT-L**"
        )
    if a >= hi and b < lo:
        return "**dc10_m05 主推**, dc08_m06 写入 sensitivity 表"
    if a < lo and b >= hi:
        return "**dc08_m06 主推**, dc10_m05 作 sensitivity 异常高点说明"
    if a < lo and b < lo:
        return "**均 <0.900** → 停, 重新规划 (勿自动 D2/D3)"
    return "部分 seed 缺失, 待补全后再判定"


def main() -> None:
    grid_rows = []
    for tag in GRID_TAGS:
        d = find_grid_dir(tag)
        f = best_f1(d) if d else None
        dc, mass = PARAMS[tag]
        grid_rows.append((tag, dc, mass, f, d.name if d else "MISSING"))

    d1_f1, d1_mean, d1_std, d1_max = multiseed_stats("dc10_m05")
    d1b_f1, d1b_mean, d1b_std, d1b_max = multiseed_stats("dc08_m06")

    m045_f = best_f1(ROOT / "outputs" / "pp_ib_partial_ot_grid_dc10_m045_seed42")

    all_f1 = [f for _, _, _, f, _ in grid_rows if f is not None]
    all_f1 += [v for v in d1_f1.values() if v is not None]
    all_f1 += [v for v in d1b_f1.values() if v is not None]
    if m045_f is not None:
        all_f1.append(m045_f)
    phase2_best = max(all_f1) if all_f1 else None

    lines = [
        "# Phase 2 — D 系列汇总 (dc10_m05 + dc08_m06 双峰)",
        "",
        "## 1. Partial OT grid (B2 + C2, seed=42)",
        "",
        "| tag | dust | mass | test F1 | dir |",
        "|---|---:|---:|---:|---|",
    ]
    for tag, dc, mass, f, dn in grid_rows:
        mark = ""
        if f and f >= 0.90:
            mark = " **≥0.90**"
        lines.append(
            f"| {tag} | {dc} | {mass} | {f'{f:.5f}' if f else '—'}{mark} | `{dn}` |"
        )

    def _seed_table(title: str, f1s: dict, mean, std, mx) -> list[str]:
        out = [f"## {title}", "", "| seed | test F1 |", "|---|---:|"]
        for s in (42, 123, 456):
            f = f1s[s]
            out.append(f"| {s} | {f'{f:.5f}' if f else '—'} |")
        if mx is not None:
            out += [
                "",
                f"**Mean** = {mean:.5f}",
                f"**Std**  = {std:.5f}",
                f"**Max**  = {mx:.5f}",
            ]
        return out

    lines += [""] + _seed_table("2. D1 — dc10_m05 三 seed", d1_f1, d1_mean, d1_std, d1_max)
    lines += [""] + _seed_table("3. D1b — dc08_m06 三 seed", d1b_f1, d1b_mean, d1b_std, d1b_max)

    if m045_f is not None:
        lines += [
            "",
            "## 4. D-extra — dc10_m045 (seed=42)",
            f"- test F1 = **{m045_f:.5f}**",
        ]

    lines += [
        "",
        "## 5. 当前 Phase 2 最强",
        f"- **全局单 seed 最高**: {phase2_best:.5f}" if phase2_best else "- **全局单 seed 最高**: —",
        f"- **dc10_m05 max-of-3**: {d1_max:.5f}" if d1_max else "- **dc10_m05 max-of-3**: —",
        f"- **dc08_m06 max-of-3**: {d1b_max:.5f}" if d1b_max else "- **dc08_m06 max-of-3**: —",
        "",
        "## 6. 主推判定 (勿自动 D2/D3)",
        "",
        "| dc10_m05 max | dc08_m06 max | 结论 |",
        "|---:|---:|---|",
        f"| {d1_max if d1_max else '—'} | {d1b_max if d1b_max else '—'} | {decision_row(d1_max, d1b_max)} |",
        "",
        "_完整 grid 亦见 `outputs/partial_ot_grid_all.md`_",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
