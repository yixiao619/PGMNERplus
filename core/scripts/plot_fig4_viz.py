#!/usr/bin/env python3
"""Plot Fig.4 visualization analysis."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from fig_common import FIG_DIR, VIZ_DIR, ensure_dirs

BLUE = "#1f4e79"
GRAY = "#bbbbbb"


def main() -> None:
    ensure_dirs()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 10,
            "axes.labelsize": 8,
        }
    )
    gates_b = np.load(VIZ_DIR / "gates_before.npy")
    gates_a = np.load(VIZ_DIR / "gates_after.npy")
    plan_full = np.load(VIZ_DIR / "plan_full.npy")
    plan_gw = np.load(VIZ_DIR / "plan_gw.npy")
    plan_partial = np.load(VIZ_DIR / "plan_partial.npy")
    tsne_b = np.load(VIZ_DIR / "tsne_before.npy")
    tsne_a = np.load(VIZ_DIR / "tsne_after.npy")
    labels = np.load(VIZ_DIR / "tsne_labels.npy", allow_pickle=True)

    gbar_after = float(gates_a.mean())
    print(f"gbar_after = {gbar_after:.2f}")

    for name, arr in [("full", plan_full), ("gw", plan_gw), ("partial", plan_partial)]:
        nz = int(np.count_nonzero(arr > 1e-6))
        print(f"OT {name}: nonzero={nz} sparsity={1-nz/arr.size:.3f}")

    colors = {"PER": "#1f77b4", "LOC": "#2ca02c", "ORG": "#ff7f0e", "MISC": "#d62728"}

    fig = plt.figure(figsize=(10.5, 3.0), dpi=300)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # (a) gate histogram
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(gates_b, bins=30, range=(0, 1), alpha=0.55, color=GRAY, label="Before")
    ax0.hist(gates_a, bins=30, range=(0, 1), alpha=0.55, color=BLUE, label="After")
    ax0.axvline(0.5, color=GRAY, linestyle="--", linewidth=0.8)
    ax0.text(
        0.98,
        0.95,
        rf"$\bar{{g}}_{{\mathrm{{after}}}}={gbar_after:.2f}$",
        transform=ax0.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    ax0.set_xlabel("Gate value")
    ax0.set_ylabel("Count")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.set_title("(a) Polarization gate distribution", y=-0.22)

    # (b) transport mini-heatmaps
    sub = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 1], wspace=0.15)
    plans = [("Full OT", plan_full), ("GW", plan_gw), ("Partial OT", plan_partial)]
    ims = []
    for i, (title, arr) in enumerate(plans):
        ax = fig.add_subplot(sub[i])
        im = ax.imshow(arr, aspect="auto", cmap="YlGnBu")
        ims.append(im)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(title, fontsize=7)
        if "Partial" in title:
            n, m = arr.shape[0] - 1, arr.shape[1] - 1
            ax.add_patch(
                plt.Rectangle(
                    (m - 0.5, -0.5),
                    1,
                    n + 1,
                    fill=False,
                    edgecolor="red",
                    linewidth=1.2,
                )
            )
            ax.add_patch(
                plt.Rectangle(
                    (-0.5, n - 0.5),
                    m + 1,
                    1,
                    fill=False,
                    edgecolor="red",
                    linewidth=1.2,
                )
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.colorbar(ims[-1], ax=[fig.axes[i] for i in range(1, 4)], fraction=0.046, pad=0.04)
    fig.axes[1].set_title("(b) Transport plans", y=1.15)

    # (c) t-SNE before/after
    sub2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 2], wspace=0.2)
    for j, (emb, title) in enumerate([(tsne_b, "Before"), (tsne_a, "After")]):
        ax = fig.add_subplot(sub2[j])
        for lab in sorted(set(labels)):
            m = labels == lab
            ax.scatter(
                emb[m, 0],
                emb[m, 1],
                s=8,
                alpha=0.6,
                c=colors.get(str(lab), "#333333"),
                label=str(lab),
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(title, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if j == 1:
            ax.legend(fontsize=6, loc="upper right", frameon=False)
    fig.axes[5].set_title("(c) Token representations (t-SNE)", y=1.12)

    out = FIG_DIR / "fig4_viz"
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}.pdf/png")


if __name__ == "__main__":
    main()
