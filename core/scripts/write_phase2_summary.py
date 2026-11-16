#!/usr/bin/env python3
"""Write outputs/phase2_summary.md from Phase 2 runs."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "phase2_summary.md"


def f1_from_dir(d: Path) -> float | None:
    log = d / "train.log"
    if log.exists():
        m = re.findall(r"best_score\s+span\s+f1\s*[=:]\s*([0-9.]+)", log.read_text(errors="ignore"), re.I)
        if m:
            return float(m[-1])
    er = d / "eval_results.txt"
    if er.exists():
        t = er.read_text(errors="ignore")
        m = re.search(
            r"\*\*\*\*\* Eval results test best ckpt \*\*\*\*\*[\s\S]*?span_f1\s*=\s*([0-9.]+)",
            t,
        )
        if m:
            return float(m.group(1))
    return None


def find_dir(pattern: str) -> Path | None:
    cands = sorted(ROOT.glob(f"outputs/{pattern}"))
    return cands[0] if cands else None


def main() -> None:
    # C1 seeds
    c1_dirs = {
        42: find_dir("pp_ib_partial_ot_grid_dc10_m06") or find_dir("pp_ib_partial_ot_grid_dc10_m06_seed42"),
        123: find_dir("pp_ib_partial_ot_grid_dc10_m06_seed123"),
        456: find_dir("pp_ib_partial_ot_grid_dc10_m06_seed456"),
    }
    c1_f1 = {s: f1_from_dir(d) if d else None for s, d in c1_dirs.items()}
    c1_vals = [v for v in c1_f1.values() if v is not None]
    c1_max = max(c1_vals) if c1_vals else None

    c2_tags = ["dc10_m05", "dc10_m055", "dc08_m06", "dc12_m06", "dc07_m065"]
    c2_rows = []
    for tag in c2_tags:
        d = find_dir(f"pp_ib_partial_ot_grid_{tag}_seed42")
        f = f1_from_dir(d) if d else None
        c2_rows.append((tag, f))
    c2_best = max((f for _, f in c2_rows if f), default=None)
    c2_best_tag = next((t for t, f in c2_rows if f == c2_best), None)

    baseline = 0.89918
    vit_d = find_dir("pp_ib_partial_ot_grid_dc10_m06_vitL14_seed42")
    ema_d = find_dir("pp_ib_partial_ot_grid_dc10_m06_ema9995_seed42")
    vit_f1 = f1_from_dir(vit_d) if vit_d else None
    ema_f1 = f1_from_dir(ema_d) if ema_d else None

    all_candidates = [
        ("B2 baseline dc10_m06 seed42", baseline, "pp_ib_partial_ot_grid_dc10_m06"),
    ]
    for s, f in c1_f1.items():
        if f is not None:
            all_candidates.append((f"C1 seed={s}", f, str(c1_dirs[s])))
    for tag, f in c2_rows:
        if f is not None:
            all_candidates.append((f"C2 {tag}", f, f"pp_ib_partial_ot_grid_{tag}_seed42"))
    if vit_f1 is not None:
        all_candidates.append(("C3 ViT-L/14", vit_f1, "pp_ib_partial_ot_grid_dc10_m06_vitL14_seed42"))
    if ema_f1 is not None:
        all_candidates.append(("C4 ema 0.9995", ema_f1, "pp_ib_partial_ot_grid_dc10_m06_ema9995_seed42"))

    best_name, best_f1, best_path = max(all_candidates, key=lambda x: x[1])

    lines = [
        "# Phase 2 汇总（B2 winning → ≥0.905）",
        "",
        "## C1 — B2 winning 多 seed (dc=1.0, mass=0.6)",
        "",
        "| seed | dir | test F1 |",
        "|---|---|---:|",
    ]
    for s in (42, 123, 456):
        d = c1_dirs[s]
        f = c1_f1[s]
        lines.append(
            f"| {s} | `{d.name if d else 'MISSING'}` | {f'{f:.5f}' if f else '—'} |"
        )
    lines += [
        "",
        f"**max-of-3** = {c1_max:.5f}" if c1_max else "**max-of-3** = —",
        f"**≥ 0.900** = {'是' if c1_max and c1_max >= 0.900 else '否'}",
        f"**≥ 0.901 (C3 gate)** = {'是' if c1_max and c1_max >= 0.901 else '否'}",
        "",
        "## C2 — 细 grid (seed=42)",
        "",
        f"Baseline **dc10_m06** = {baseline:.5f}",
        "",
        "| tag | dust | mass | test F1 |",
        "|---|---:|---:|---:|",
    ]
    params = {
        "dc10_m05": (1.0, 0.5),
        "dc10_m055": (1.0, 0.55),
        "dc08_m06": (0.8, 0.6),
        "dc12_m06": (1.2, 0.6),
        "dc07_m065": (0.7, 0.65),
    }
    for tag, f in c2_rows:
        dc, mass = params[tag]
        lines.append(f"| {tag} | {dc} | {mass} | {f'{f:.5f}' if f else '—'} |")
    lines += [
        "",
        f"**C2 best** = {c2_best:.5f} ({c2_best_tag})" if c2_best else "**C2 best** = —",
        f"**新峰 ≥ 0.901** = {'是 → 建议用 ' + str(c2_best_tag) + ' 重做 3 seed' if c2_best and c2_best >= 0.901 else '否'}",
        "",
        "## C3 / C4（条件性）",
        "",
        f"- **C3 ViT-L/14**: {vit_f1 if vit_f1 is not None else '（未跑或跳过）'}",
        f"- **C4 ema 0.9995**: {ema_f1 if ema_f1 is not None else '（未跑或跳过）'}",
        "",
        "## 当前最强",
        f"- **配置**: {best_name}",
        f"- **F1**: {best_f1:.5f}",
        f"- **目录**: `{best_path}`",
        "",
        "## 判定",
    ]
    if best_f1 >= 0.905:
        lines.append("- ✅ 单 seed 已达 **≥0.905**，可进多 seed 确认 + 论文主表。")
    elif best_f1 >= 0.901:
        lines.append("- 接近目标；优先 C4 / 最优 C2 cell 的 3-seed，或 C3 ViT-L。")
    else:
        lines.append("- 未达 0.905；继续扫 partial 超参或 ensemble。")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
