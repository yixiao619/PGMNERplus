#!/usr/bin/env python3
"""Plot Fig.3 sensitivity from results/fig3_sweep/*.csv"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fig3_sweep"

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)


def load(p: Path):
    xs, ys = [], []
    with p.open() as f:
        for r in csv.DictReader(f):
            v = list(r.values())
            xs.append(float(v[0]))
            ys.append(float(v[1]))
    return xs, ys


def main() -> None:
    bx, by = load(OUT / "beta.csv")
    sx, sy = load(OUT / "sinit.csv")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    plt.subplots_adjust(wspace=0.35)

    a1.plot(bx, by, "-o", color="#2c7fb8", ms=4, lw=1.4)
    a1.set_xscale("log")
    a1.set_xlabel(r"$\beta_{\mathrm{polar}}$")
    a1.set_ylabel("Test F1")
    a1.set_title(r"(a) Sensitivity to $\beta_{\mathrm{polar}}$", fontsize=9)
    if by:
        bi = by.index(max(by))
        a1.scatter(
            [bx[bi]], [by[bi]], s=60, facecolors="none", edgecolors="#d95f0e", lw=1.5, zorder=5
        )

    a2.plot(sx, sy, "-s", color="#238b45", ms=4, lw=1.4)
    a2.set_xlabel(r"$s_{\mathrm{init}}$ (partial-OT mass)")
    a2.set_ylabel("Test F1")
    a2.set_title(r"(b) Sensitivity to $s_{\mathrm{init}}$", fontsize=9)
    if sy:
        si = sy.index(max(sy))
        a2.scatter(
            [sx[si]], [sy[si]], s=60, facecolors="none", edgecolors="#d95f0e", lw=1.5, zorder=5
        )

    for ax in (a1, a2):
        ax.grid(True, ls=":", lw=0.5, alpha=0.5)

    for ext in ("png", "pdf"):
        path = ROOT / f"fig3_sensitivity.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"saved {path}")


if __name__ == "__main__":
    main()
