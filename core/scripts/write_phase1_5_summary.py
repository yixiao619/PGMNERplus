#!/usr/bin/env python3
"""Write outputs/phase1_5_summary.md from experiment dirs + decision flags."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_MD = ROOT / "outputs" / "phase1_5_summary.md"


def best_f1(out_dir: Path) -> float | None:
    log = out_dir / "train.log"
    if log.exists():
        m = re.findall(r"best_score\s+span\s+f1\s*[=:]\s*([0-9.]+)", log.read_text(errors="ignore"), re.I)
        if m:
            return float(m[-1])
    er = out_dir / "eval_results.txt"
    if er.exists():
        t = er.read_text(errors="ignore")
        m = re.search(
            r"\*\*\*\*\* Eval results test best ckpt \*\*\*\*\*[\s\S]*?span_f1\s*=\s*([0-9.]+)",
            t,
        )
        if m:
            return float(m.group(1))
    return None


def main() -> None:
    b1 = best_f1(ROOT / "outputs" / "pp_ib_full_ot_seed42")
    b3 = best_f1(ROOT / "outputs" / "pp_ib_lrtune_seed42")
    b4 = best_f1(ROOT / "outputs" / "pp_ib_vitL14_seed42")
    a4 = best_f1(ROOT / "outputs" / "pp_ib_partial_ot_ema_focal_seed42")
    baseline = 0.89578

    if b1 is not None and b1 >= 0.895:
        ot_decision = "OT 机制可保留 → 可跑 B2 partial 微 grid"
    elif b1 is not None and b1 < 0.890:
        ot_decision = "OT（full/partial）在 PP-IB 后偏冗余 → 论文 OT 作 ablation"
    else:
        ot_decision = "待 B1 结果"

    grid_rows = []
    for d in sorted((ROOT / "outputs").glob("pp_ib_partial_ot_grid_*")):
        f = best_f1(d)
        if f is not None:
            grid_rows.append((d.name, f))
    best_grid = max((f for _, f in grid_rows), default=None)

    candidates = [
        ("PP-IB only (task6)", baseline, "pp_ib_fixed_b16 / seed42"),
        ("A4 focal+partial OT", a4, "pp_ib_partial_ot_ema_focal_seed42"),
        ("B1 full OT + EMA", b1, "pp_ib_full_ot_seed42"),
        ("B3 lr tune + EMA", b3, "pp_ib_lrtune_seed42"),
        ("B4 ViT-L/14 + B3", b4, "pp_ib_vitL14_seed42"),
    ]
    if best_grid is not None:
        candidates.append(("B2 best partial grid", best_grid, "pp_ib_partial_ot_grid_*"))

    valid = [(n, f, p) for n, f, p in candidates if f is not None]
    best_name, best_f, best_path = max(valid, key=lambda x: x[1]) if valid else ("—", None, "—")

    lines = [
        "# Phase 1.5 判定实验汇总",
        "",
        "## A4（Phase 1 收尾）",
        f"- **best test span F1**: {a4 if a4 is not None else '（未完成）'}",
        "",
        "## B1 — Full OT 对照",
        f"- **best F1**: {b1 if b1 is not None else '（未跑）'}",
        f"- **Decision**: {ot_decision}",
        "",
        "## B3 — 训练超参基线",
        f"- **best F1**: {b3 if b3 is not None else '（未跑）'}",
        f"- **≥ 0.90**: {'是' if b3 and b3 >= 0.90 else '否' if b3 else '—'}",
        "",
        "## B4 — ViT-L/14",
        f"- **best F1**: {b4 if b4 is not None else '（未跑）'}",
        f"- **≥ 0.905**: {'是' if b4 and b4 >= 0.905 else '否' if b4 else '—'}",
        "",
    ]
    if grid_rows:
        lines += ["## B2 — Partial OT 微 grid", "", "| run | F1 |", "|---|---|"]
        for name, f in grid_rows:
            lines.append(f"| {name} | {f:.5f} |")
        lines.append("")

    lines += [
        "## 当前最强单 seed",
        f"- **配置**: {best_name}",
        f"- **F1**: {best_f:.5f}" if best_f else "- **F1**: —",
        f"- **目录**: `{best_path}`",
        "",
        "## Phase 2 推荐起点",
        "- PP-IB + EMA + `bert_lr=1e-5`, `batch_size=16`, `warmup_ratio=0.1`, `epoch=40`",
        "- 视觉：优先 ViT-L/14（若 B4 最优）",
        "- OT：按 B1 decision；partial 仅 B1≥0.895 且 B2 grid 有增益时启用",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
