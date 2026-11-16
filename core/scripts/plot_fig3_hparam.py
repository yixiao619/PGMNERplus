#!/usr/bin/env python3
"""Plot Fig.3 hyperparameter sensitivity from results/hparam_sweep.jsonl."""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib.pyplot as plt
import numpy as np

from fig_common import BETA_VALUES, FIG_DIR, JSONL, SINIT_VALUES, ensure_dirs, read_jsonl

BLUE = "#1f4e79"
GRAY = "#888888"
PEAK_BETA = 0.01


def filter_legacy_seed42(rows):
    out = []
    for r in rows:
        src = str(r.get("source", ""))
        if r.get("seed") == 42 and src.startswith("legacy"):
            out.append(r)
    return out


def aggregate(rows, name: str):
    by_val: dict = defaultdict(list)
    for r in rows:
        if r["hparam_name"] != name:
            continue
        if r.get("best_f1") is None:
            continue
        by_val[float(r["value"])].append(float(r["best_f1"]))
    vals = sorted(by_val.keys())
    means, stds, ns = [], [], []
    for v in vals:
        xs = by_val[v]
        means.append(statistics.mean(xs) * 100.0)
        stds.append(statistics.stdev(xs) if len(xs) > 1 else 0.0)
        ns.append(len(xs))
    return vals, means, stds, ns


def peak_from_series(vals, means):
    pairs = [(v, m) for v, m in zip(vals, means) if np.isfinite(m)]
    if not pairs:
        return None, np.nan
    v, m = max(pairs, key=lambda x: x[1])
    return v, m


def annotate_peak(ax, x, y, xtext=None, ytext=None):
    ax.plot(
        x,
        y,
        marker="*",
        markersize=12,
        markerfacecolor="black",
        markeredgecolor="black",
        zorder=6,
    )
    ax.annotate(
        f"{y:.2f}",
        xy=(x, y),
        xytext=xtext or (x, y + 0.08),
        ha="center",
        fontsize=8,
        fontweight="bold",
        color="black",
    )


def plot(rows, out_prefix: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), dpi=300)

    # (a) beta_polar
    ax = axes[0]
    vals, means, stds, ns = aggregate(rows, "beta_polar")
    if not vals:
        vals = BETA_VALUES
        means = [np.nan] * len(vals)
        stds = [0.0] * len(vals)
    ax.errorbar(
        vals,
        means,
        yerr=stds,
        fmt="o-",
        color="black",
        ecolor=GRAY,
        capsize=3,
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.0,
        linewidth=1.2,
    )
    ax.set_xscale("log")
    ax.set_xticks(BETA_VALUES)
    ax.set_xticklabels([str(v) for v in BETA_VALUES])
    ax.set_xlabel(r"Polarization weight $\beta_{\mathrm{polar}}$")
    ax.set_ylabel("Test F1 (%)")
    if any(np.isfinite(means)):
        ymin = min(m for m in means if np.isfinite(m))
        ymax = max(m for m in means if np.isfinite(m))
        pad = max(0.15, (ymax - ymin) * 0.35)
        ax.set_ylim(ymin - pad, ymax + pad + 0.12)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    peak_x, peak_y = peak_from_series(vals, means)
    if peak_x is not None and np.isfinite(peak_y):
        annotate_peak(ax, peak_x, peak_y)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("(a) $\\beta_{\\mathrm{polar}}$", y=-0.28)

    # (b) s_init
    ax = axes[1]
    vals2, means2, stds2, ns2 = aggregate(rows, "s_init")
    if not vals2:
        vals2 = SINIT_VALUES
        means2 = [np.nan] * len(vals2)
        stds2 = [0.0] * len(vals2)
    ax.errorbar(
        vals2,
        means2,
        yerr=stds2,
        fmt="o-",
        color="black",
        ecolor=GRAY,
        capsize=3,
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.0,
        linewidth=1.2,
    )
    ax.set_xlabel(r"Initial mass $s_{\mathrm{init}}$")
    ax.set_ylabel("Test F1 (%)")
    if any(np.isfinite(means2)):
        ymin = min(m for m in means2 if np.isfinite(m))
        ymax = max(m for m in means2 if np.isfinite(m))
        pad = max(0.15, (ymax - ymin) * 0.35)
        ax.set_ylim(ymin - pad, ymax + pad + 0.12)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    peak_x, peak_y = peak_from_series(vals2, means2)
    if peak_x is not None and np.isfinite(peak_y):
        annotate_peak(ax, peak_x, peak_y)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("(b) $s_{\\mathrm{init}}$", y=-0.28)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

    print("\n=== Fig.3 mean ± std (F1 %) ===")
    for name, ref_vals in [("beta_polar", BETA_VALUES), ("s_init", SINIT_VALUES)]:
        vals, means, stds, ns = aggregate(rows, name)
        print(f"\n[{name}]")
        for v in ref_vals:
            if v in vals:
                i = vals.index(v)
                print(f"  {v}: {means[i]:.2f} ± {stds[i]:.2f}  (n={ns[i]})")
            else:
                print(f"  {v}: — (no data)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=JSONL)
    parser.add_argument("--out", type=Path, default=FIG_DIR / "fig3_hparam")
    parser.add_argument(
        "--legacy-only",
        action="store_true",
        help="Use only legacy seed=42 rows (40ep, dust=0.8, s_init=0.6).",
    )
    args = parser.parse_args()
    ensure_dirs()
    rows = read_jsonl(args.jsonl)
    if args.legacy_only:
        rows = filter_legacy_seed42(rows)
        if not rows:
            raise SystemExit("No legacy seed=42 rows found in jsonl.")
    plot(rows, args.out)
    print(f"\nWrote {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()
