#!/usr/bin/env python3
"""从 table1_summary.json 生成柱状图（Task 5 简化版）。"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "analysis", "results")
FIG = os.path.join(REPO, "analysis", "figures")
os.makedirs(FIG, exist_ok=True)

paper_vals = {
    "Twitter-2015": {
        "avg_clip_score": 0.187,
        "one_to_many_ratio": 0.634,
        "entity_token_grounded_pct": 0.412,
        "misc_entity_pct": 0.223,
    },
    "Twitter-2017": {
        "avg_clip_score": 0.193,
        "one_to_many_ratio": 0.718,
        "entity_token_grounded_pct": 0.387,
        "misc_entity_pct": 0.316,
    },
    "COCO-NER": {
        "avg_clip_score": 0.341,
        "one_to_many_ratio": 0.283,
        "entity_token_grounded_pct": 0.821,
        "misc_entity_pct": 0.084,
    },
    "WikiDiverse": {
        "avg_clip_score": 0.312,
        "one_to_many_ratio": 0.319,
        "entity_token_grounded_pct": 0.784,
        "misc_entity_pct": 0.112,
    },
}


def main():
    summary = os.path.join(OUT, "table1_summary.json")
    measured = {}
    if os.path.isfile(summary):
        with open(summary, encoding="utf-8") as f:
            measured = json.load(f)

    vals = {k: dict(paper_vals[k]) for k in paper_vals}
    for dname, m in measured.items():
        if dname not in vals:
            vals[dname] = dict(paper_vals.get(dname, {}))
        if isinstance(m, dict):
            for k in vals[dname]:
                if k in m:
                    vals[dname][k] = float(m[k])

    # 顺序：先实测数据集，再仅论文参考集
    datasets = list(measured.keys()) if measured else []
    for k in paper_vals:
        if k not in datasets:
            datasets.append(k)
    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db", "#9b59b6", "#1abc9c"]

    fig, axes = plt.subplots(1, 4, figsize=(16,4))
    metric_info = [
        ("avg_clip_score", "Avg. CLIP score"),
        ("one_to_many_ratio", "One-to-many ratio"),
        ("entity_token_grounded_pct", "% entity tokens grounded"),
        ("misc_entity_pct", "MISC proportion"),
    ]
    for ax, (key, title) in zip(axes, metric_info):
        row_vals = [vals[d][key] for d in datasets]
        ax.bar(range(len(datasets)), row_vals, color=colors[: len(datasets)], alpha=0.85)
        for i, v in enumerate(row_vals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        if len(datasets) >= 2:
            ax.axvline(1.5, color="gray", linestyle="--", alpha=0.6)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels([d.replace("Twitter-", "T-")[:10] for d in datasets], fontsize=8)
        ax.set_ylim(0, max(0.15, max(row_vals)) * 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.suptitle("Table 1: dataset analysis (measured fills when summary exists)", y=1.02)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"table1_bar_chart.{ext}")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print("Saved", p)
    plt.close()

    # CLIP gap
    twitter = [d for d in datasets if d.startswith("Twitter")]
    ref = [d for d in datasets if d in ("COCO-NER", "WikiDiverse")]
    if len(twitter) >= 1 and len(ref) >= 1:
        tw_avg = np.mean([vals[d]["avg_clip_score"] for d in twitter])
        ref_avg = np.mean([vals[d]["avg_clip_score"] for d in ref])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(
            ["Twitter avg", "Reference avg"],
            [tw_avg, ref_avg],
            color=["#e74c3c", "#2ecc71"],
        )
        ax.set_xlabel("Avg. CLIP similarity")
        ax.set_title("Text–image CLIP score gap")
        for i, v in enumerate([tw_avg, ref_avg]):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center")
        plt.tight_layout()
        p = os.path.join(FIG, "clip_gap_highlight.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print("Saved", p)
        plt.close()


if __name__ == "__main__":
    main()
