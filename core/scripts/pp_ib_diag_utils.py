#!/usr/bin/env python3
"""PP-IB polarization diagnostics (gate / mu / verdict)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm


def _bimodality(gates: np.ndarray) -> float:
    g = gates.reshape(-1)
    if g.size == 0:
        return 0.0
    near_extreme = np.mean((g < 0.15) | (g > 0.85))
    return float(near_extreme)


def _frac_mid(gates: np.ndarray, lo: float = 0.4, hi: float = 0.6) -> float:
    g = gates.reshape(-1)
    if g.size == 0:
        return 1.0
    return float(np.mean((g >= lo) & (g <= hi)))


@torch.no_grad()
def collect_pp_ib_stats(model, loader, device) -> Dict[str, float]:
    gates: List[np.ndarray] = []
    mu_norms: List[np.ndarray] = []
    logit_stds: List[float] = []

    model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        _, txt_rep_g, _ = model.span_encoding(**batch)
        vis_g, _ = model.image_model(batch["image"])

        for dist, rep in ((model.dist_t, txt_rep_g), (model.dist_v, vis_g)):
            mu = dist.mu_net(rep)
            gate_in = rep if getattr(dist, "gate_on_hidden", False) else mu
            lin = dist.polar_gate[0]
            logits = lin(gate_in).squeeze(-1)
            gamma = torch.sigmoid(logits)
            gates.append(gamma.detach().cpu().numpy().reshape(-1))
            mu_norms.append(mu.detach().norm(dim=-1).cpu().numpy().reshape(-1))
            logit_stds.append(float(logits.std().item()))

    g = np.concatenate(gates) if gates else np.array([], dtype=np.float32)
    mu = np.concatenate(mu_norms) if mu_norms else np.array([], dtype=np.float32)
    stats = {
        "gate_mean": float(g.mean()) if g.size else 0.5,
        "gate_std": float(g.std()) if g.size else 0.0,
        "bimodality": _bimodality(g),
        "frac_mid": _frac_mid(g),
        "mu_norm_mean": float(mu.mean()) if mu.size else 0.0,
        "mu_norm_std": float(mu.std()) if mu.size else 0.0,
        "logit_std_mean": float(np.mean(logit_stds)) if logit_stds else 0.0,
        "n_gates": int(g.size),
    }
    stats["gates"] = g
    return stats


def _verdict(stats: Dict[str, Any]) -> str:
    mu_norm = float(stats.get("mu_norm_mean", 0.0))
    gate_std = float(stats.get("gate_std", 0.0))
    gate_mean = float(stats.get("gate_mean", 0.5))
    bimodality = float(stats.get("bimodality", 0.0))
    frac_mid = float(stats.get("frac_mid", 1.0))
    logit_std = float(stats.get("logit_std_mean", 0.0))

    if mu_norm < 0.05:
        return "VIB-collapse"
    if mu_norm < 0.3 and gate_std < 0.03 and abs(gate_mean - 0.5) < 0.05:
        return "VIB-collapse"
    if gate_std < 0.02 and logit_std < 0.05 and abs(gate_mean - 0.5) < 0.05:
        return "Logit-stuck"
    if bimodality < 0.35 and frac_mid > 0.75:
        if abs(gate_mean - 0.5) < 0.04:
            return "balance dominates"
        return "beta too weak"
    if abs(gate_mean - 0.5) < 0.03 and frac_mid > 0.85 and bimodality < 0.25:
        return "balance dominates"
    return "OK"


def plot_diag(
    history: List[Dict[str, Any]],
    out_path: Path,
    title: str = "",
    verdict: Optional[str] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    last_gates = history[-1].get("gates", np.array([]))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_gate, ax_bimod, ax_mu, ax_mid = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    if last_gates.size:
        ax_gate.hist(last_gates, bins=40, color="#4472C4", alpha=0.85, edgecolor="white")
        ax_gate.axvline(0.5, color="red", ls="--", lw=1)
        ax_gate.set_xlabel("Gate value")
        ax_gate.set_ylabel("Count")
        ax_gate.set_title(f"Gate histogram (epoch {epochs[-1]})")
    else:
        ax_gate.text(0.5, 0.5, "no gates", ha="center", va="center")

    ax_bimod.plot(epochs, [h["bimodality"] for h in history], "o-", color="#ED7D31")
    ax_bimod.axhline(0.6, color="green", ls=":", lw=1, label="target 0.6")
    ax_bimod.set_xlabel("Epoch")
    ax_bimod.set_ylabel("Bimodality")
    ax_bimod.set_ylim(0, 1)
    ax_bimod.legend(fontsize=8)
    ax_bimod.set_title("Bimodality (frac near 0 or 1)")

    ax_mu.plot(epochs, [h["mu_norm_mean"] for h in history], "o-", color="#70AD47")
    ax_mu.set_xlabel("Epoch")
    ax_mu.set_ylabel(r"$||\mu||$ mean")
    ax_mu.set_title(r"Text/visual $\mu$ norm")

    ax_mid.plot(epochs, [h["frac_mid"] for h in history], "o-", color="#FFC000")
    ax_mid.axhline(0.2, color="green", ls=":", lw=1, label="target <0.2")
    ax_mid.set_xlabel("Epoch")
    ax_mid.set_ylabel("frac_mid")
    ax_mid.set_ylim(0, 1)
    ax_mid.legend(fontsize=8)
    ax_mid.set_title("Fraction of gates in [0.4, 0.6]")

    subtitle = title or "PP-IB diagnostic"
    if verdict:
        subtitle += f"  |  verdict: {verdict}"
    fig.suptitle(subtitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def save_history_json(history: List[Dict[str, Any]], path: Path) -> None:
    slim = []
    for h in history:
        row = {k: v for k, v in h.items() if k != "gates"}
        slim.append(row)
    path.write_text(json.dumps(slim, indent=2), encoding="utf-8")


def make_dev_loader(examples, args, tokenizer, labels, collate_fn):
    from utils.datasets import SpanNerDataset

    ds = SpanNerDataset(examples, args=args, tokenizer=tokenizer, labels=labels, dev=True)
    return DataLoader(
        ds,
        sampler=SequentialSampler(ds),
        batch_size=min(16, args.batch_size),
        collate_fn=collate_fn,
    )
