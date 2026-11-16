#!/usr/bin/env python3
"""Motivation figure: conventional OT vs PGMNER partial transport (toy heatmaps)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "figures"

tokens = ["Marvel", "rewatch", "weekend", "popcorn", "ready"]
patches = ["R1 popcorn", "R2 sofa", "R3 blanket", "R4 lamp"]
rng = np.random.default_rng(0)

# (a) diffuse + column-normalized (F1/F2 failure modes)
A = 0.35 + 0.15 * rng.random((len(patches), len(tokens)))
A = A / A.sum(axis=0, keepdims=True)

# (b) near-binary + dust-bin row
rows_b = patches + ["dust-bin"]
B = np.zeros((len(rows_b), len(tokens)))
B[0, tokens.index("popcorn")] = 0.9
for t in ["Marvel", "rewatch", "weekend", "ready"]:
    B[-1, tokens.index(t)] = 0.9
B += 0.03 * rng.random(B.shape)

CMAP = "Blues"
VMIN, VMAX = 0.0, 1.0

fig, ax = plt.subplots(2, 1, figsize=(3.4, 4.2), constrained_layout=True)
ims = []
for a, M, rows, ttl in [
    (ax[0], A, patches, "(a) Conventional pipeline"),
    (ax[1], B, rows_b, "(b) PGMNER (ours)"),
]:
    im = a.imshow(M, cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect="auto")
    ims.append(im)
    a.set_xticks(range(len(tokens)))
    a.set_xticklabels(tokens, rotation=30, ha="right", fontsize=7)
    a.set_yticks(range(len(rows)))
    a.set_yticklabels(rows, fontsize=7)
    a.set_title(ttl, fontsize=8)
    a.set_ylabel("patches", fontsize=7)

ax[1].set_xlabel("text tokens", fontsize=7)
fig.colorbar(ims[-1], ax=ax, shrink=0.7, label="affinity / mass")

OUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    out = OUT_DIR / f"motivation.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=300 if ext == "png" else None)
    print(f"saved {out}")
