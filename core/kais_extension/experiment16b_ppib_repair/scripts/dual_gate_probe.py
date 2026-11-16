"""Training-only PP-IB probe that records text and visual gates separately."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch


class DualGateProbe:
    def __init__(self, out_dir, beta_polar=0.01, lambda_balance=1.0):
        self.out_dir = out_dir
        self.beta_polar = float(beta_polar)
        self.lambda_balance = float(lambda_balance)
        os.makedirs(out_dir, exist_ok=True)
        self.history = []
        self._reset()

    def _reset(self):
        self.text = []
        self.visual = []

    @torch.no_grad()
    def log_step(self, mu, gate, gate_logit, mask=None):
        g = gate.detach().float().cpu()
        if g.ndim != 2 or g.shape[1] != 2:
            raise ValueError(f"expected combined gate [B,2], got {tuple(g.shape)}")
        self.text.append(g[:, 0].numpy())
        self.visual.append(g[:, 1].numpy())

    @staticmethod
    def _stats(values):
        g = np.concatenate(values).astype(np.float64)
        eps = 1e-12
        h = -(g * np.log(np.clip(g, eps, 1)) + (1-g) * np.log(np.clip(1-g, eps, 1))) / math.log(2)
        return {
            "n": int(g.size), "mean": float(g.mean()), "std": float(g.std()),
            "min": float(g.min()), "max": float(g.max()),
            "fraction_low_0.1": float(np.mean(g <= .1)),
            "fraction_mid_0.1_0.9": float(np.mean((g > .1) & (g < .9))),
            "fraction_high_0.9": float(np.mean(g >= .9)),
            "polarization_ratio": float(np.mean((g <= .1) | (g >= .9))),
            "normalized_entropy": float(h.mean()),
            "mean_polar_term": float(np.mean(g * (1-g))),
        }

    def epoch_end(self, epoch):
        if not self.text or not self.visual:
            return
        row = {"epoch": int(epoch), "text": self._stats(self.text), "visual": self._stats(self.visual)}
        self.history.append(row)
        print("[dual-gate-probe] " + json.dumps(row, sort_keys=True), flush=True)
        self._reset()

    def finalize(self):
        with open(os.path.join(self.out_dir, "dual_gate_history.json"), "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        return "RECORDED"
