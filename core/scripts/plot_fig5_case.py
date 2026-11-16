#!/usr/bin/env python3
"""Plot Fig.5 case study."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.image import imread

from fig_common import FIG_DIR, ROOT, VIZ_DIR, ensure_dirs


def cell_color(gold, pred):
    if gold == pred:
        return "#d5f5d5"
    return "#f8d7da"


def main() -> None:
    ensure_dirs()
    data = json.loads((VIZ_DIR / "case_study.json").read_text())
    cases = data["cases"]
    fig = plt.figure(figsize=(10.5, 5.5), dpi=300)
    gs = gridspec.GridSpec(len(cases), 4, figure=fig, wspace=0.25, hspace=0.35)

    for r, case in enumerate(cases):
        fig.text(
            0.02,
            0.78 - r * 0.48,
            case["name"],
            rotation=90,
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        # col0 image
        ax_img = fig.add_subplot(gs[r, 0])
        img_path = ROOT / case["image_path"]
        if img_path.exists():
            ax_img.imshow(imread(img_path))
        ax_img.axis("off")

        # col1 token table
        ax_txt = fig.add_subplot(gs[r, 1])
        ax_txt.axis("off")
        tokens = case["tokens"]
        gold = case["gold"]
        models = ["HVPNet", "SMNER", "PGMNER"]
        lines = ["Token (gold) | " + " | ".join(models)]
        for i, tok in enumerate(tokens[: min(len(tokens), 20)]):
            cols = []
            for m in models:
                p = case["preds"][m][i] if i < len(case["preds"][m]) else "O"
                cols.append(p)
            lines.append(f"{tok} ({gold[i]}) | " + " | ".join(cols))
        y = 1.0
        for i, line in enumerate(lines):
            if i == 0:
                ax_txt.text(0, y, line, fontsize=6, family="monospace", fontweight="bold")
                y -= 0.05
                continue
            tok_i = i - 1
            g = gold[tok_i]
            x = 0
            ax_txt.text(x, y, f"{tokens[tok_i]} ({g})", fontsize=6, family="monospace")
            x += 0.35
            for m in models:
                p = case["preds"][m][tok_i]
                ax_txt.text(
                    x,
                    y,
                    p,
                    fontsize=6,
                    family="monospace",
                    bbox=dict(facecolor=cell_color(g, p), edgecolor="none", pad=1),
                )
                x += 0.18
            y -= 0.045

        # col2 OT heatmap
        ax_ot = fig.add_subplot(gs[r, 2])
        plan = np.load(ROOT / case["ot_plan"])
        ax_ot.imshow(plan, aspect="auto", cmap="YlGnBu")
        n, m = plan.shape[0] - 1, plan.shape[1] - 1
        ax_ot.axhline(n - 0.5, color="white", linestyle="--", linewidth=0.8)
        ax_ot.axvline(m - 0.5, color="white", linestyle="--", linewidth=0.8)
        ax_ot.set_xticks([])
        ax_ot.set_yticks([])
        ax_ot.set_title("Partial OT", fontsize=8)

        # col3 dust-bin bar
        ax_bar = fig.add_subplot(gs[r, 3])
        mass = case["dustbin_mass_per_token"][: len(tokens)]
        ax_bar.bar(range(len(mass)), mass, color="#1f4e79")
        ax_bar.set_xlabel("Token idx", fontsize=7)
        ax_bar.set_ylabel("Dust mass", fontsize=7)
        ax_bar.tick_params(labelsize=6)

    out = FIG_DIR / "fig5_case"
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}.pdf/png")


if __name__ == "__main__":
    main()
