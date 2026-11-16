#!/usr/bin/env python3
"""Build the fixed Experiment 19 tables and figures from recorded artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXP = Path(__file__).resolve().parents[1]
if not (EXP / "results").exists():
    EXP = Path("/home/jyx/yixiao/MINER-My/kais_extension/experiment19_pc_ppib")
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
FIGURES.mkdir(exist_ok=True)


def main() -> None:
    learned = pd.read_csv(RESULTS / "dev_epoch_metrics_learned_gate.csv")
    fixed = pd.read_csv(RESULTS / "dev_epoch_metrics_fixed_gate_0.5.csv")
    epochs = pd.concat([learned, fixed], ignore_index=True)
    epochs.to_csv(RESULTS / "dev_epoch_metrics.csv", index=False)

    learned_gates = pd.read_csv(RESULTS / "gate_statistics_learned_gate.csv")
    fixed_gates = pd.read_csv(RESULTS / "gate_statistics_fixed_gate_0.5.csv")
    gates = pd.concat([learned_gates, fixed_gates], ignore_index=True)
    gates.to_csv(RESULTS / "gate_statistics.csv", index=False)

    learned_summary = json.loads((RESULTS / "learned_gate_summary.json").read_text())
    fixed_summary = json.loads((RESULTS / "fixed_gate_0.5_summary.json").read_text())
    residual = json.loads((RESULTS / "residual_effect.json").read_text())
    comparison = pd.DataFrame([
        {
            "model": "Original PGMNER checkpoint", "gate_mode": "original_disconnected",
            "best_dev_f1": 0.89462, "delta_dev_vs_original": 0.0, "best_epoch": "historical",
            "test_f1": 0.90647, "test_evaluated_in_exp19": False,
            "task_gate_gradient_mean": 0.0, "dev_residual_ratio": 0.0,
            "dev_changed_samples_residual_off": 0,
            "selection_note": "Historical checkpoint selected by TEST span F1",
        },
        {
            "model": "PC-PP-IB Learned Gate", "gate_mode": "learned",
            "best_dev_f1": learned_summary["best_dev_f1"],
            "delta_dev_vs_original": learned_summary["delta_dev_vs_checkpoint"],
            "best_epoch": learned_summary["best_epoch"], "test_f1": "",
            "test_evaluated_in_exp19": False,
            "task_gate_gradient_mean": learned["task_gate_grad_l2"].mean(),
            "dev_residual_ratio": residual["mean_residual_ratio"],
            "dev_changed_samples_residual_off": residual["changed_samples"],
            "selection_note": "DEV only; early stopping patience=2",
        },
        {
            "model": "PC-PP-IB Fixed Gate=0.5", "gate_mode": "fixed_0.5",
            "best_dev_f1": fixed_summary["best_dev_f1"],
            "delta_dev_vs_original": fixed_summary["delta_dev_vs_checkpoint"],
            "best_epoch": fixed_summary["best_epoch"], "test_f1": "",
            "test_evaluated_in_exp19": False,
            "task_gate_gradient_mean": fixed["task_gate_grad_l2"].mean(),
            "dev_residual_ratio": fixed_gates.loc[
                fixed_gates["epoch"] == fixed_summary["best_epoch"], "residual_ratio_mean"].mean(),
            "dev_changed_samples_residual_off": "not_run",
            "selection_note": "Retrained matched control; DEV only",
        },
    ])
    comparison.to_csv(RESULTS / "experiment_comparison.csv", index=False)

    colors = {"learned_gate": "#2364AA", "fixed_gate_0.5": "#D1495B"}
    labels = {"learned_gate": "Learned gate", "fixed_gate_0.5": "Fixed gate = 0.5"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for variant, group in epochs.groupby("variant"):
        ax.plot(group["epoch"], group["dev_f1"] * 100, marker="o", lw=2,
                color=colors[variant], label=labels[variant])
    ax.axhline(89.462, color="#333333", ls="--", lw=1.4, label="Original checkpoint")
    ax.set(xlabel="Fine-tuning epoch", ylabel="Twitter-2017 DEV F1 (%)", xticks=[1, 2, 3])
    ax.set_ylim(89.40, 89.52)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig19a_dev_f1.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True)
    for ax, modality in zip(axes, ["text", "visual"]):
        subset = gates[gates["modality"] == modality]
        for variant, group in subset.groupby("variant"):
            ax.errorbar(group["epoch"], group["mean"], yerr=group["std"], marker="o",
                        capsize=3, lw=1.8, color=colors[variant], label=labels[variant])
        ax.axhline(0.5, color="#555555", ls="--", lw=1)
        ax.set_title(f"{modality.capitalize()} gate")
        ax.set_xlabel("Fine-tuning epoch")
        ax.set_xticks([1, 2, 3])
    axes[0].set_ylabel("Gate mean (error bar: SD)")
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig19b_gate_behavior.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for variant, group in epochs.groupby("variant"):
        axes[0].plot(group["epoch"], group["task_gate_grad_l2"], marker="o", lw=1.8,
                     color=colors[variant], label=labels[variant])
        axes[1].plot(group["epoch"], group["residual_ratio_train_mean"] * 100, marker="o", lw=1.8,
                     color=colors[variant], label=labels[variant])
    axes[0].set(xlabel="Fine-tuning epoch", ylabel="Task-to-gate gradient L2", yscale="log")
    axes[1].set(xlabel="Fine-tuning epoch", ylabel="Residual / original norm (%)")
    for ax in axes:
        ax.set_xticks([1, 2, 3])
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig19c_gradient_and_residual.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
