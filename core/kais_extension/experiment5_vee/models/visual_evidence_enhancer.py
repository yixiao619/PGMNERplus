"""Complementary Visual Evidence Enhancement modules for Experiment 5.

This module is analysis/extension code only. It does not replace PP-IB, GOT-A,
Partial OT, or the original span classifier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _masked_softmax(scores: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return torch.softmax(scores, dim=-1)
    mask = mask.to(dtype=torch.bool, device=scores.device)
    while mask.dim() < scores.dim():
        mask = mask.unsqueeze(1)
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return torch.softmax(scores, dim=-1)


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=1)
    mask = mask.to(dtype=values.dtype, device=values.device).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (values * mask).sum(dim=1) / denom


@dataclass
class VEEOutput:
    hidden: torch.Tensor
    delta: torch.Tensor
    gate: torch.Tensor
    evidence: torch.Tensor
    attention: Optional[torch.Tensor]


class VisualEvidenceEnhancer(nn.Module):
    """Single shared VEE framework with config-selected variants."""

    VALID_TYPES = {"none", "mean", "attention", "gated_attention"}

    def __init__(
        self,
        hidden_dim: int,
        vee_type: str = "none",
        initial_gate: float = 0.1,
        residual_scale: float = 1.0,
        attention_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        vee_type = str(vee_type or "none").lower()
        if vee_type not in self.VALID_TYPES:
            raise ValueError(f"unknown vee_type={vee_type!r}; expected {sorted(self.VALID_TYPES)}")
        self.hidden_dim = int(hidden_dim)
        self.vee_type = vee_type
        self.residual_scale = float(residual_scale)
        self.attention_temperature = float(attention_temperature)

        if self.vee_type == "none":
            return

        if self.vee_type == "mean":
            self.evidence_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.alpha_logit = nn.Parameter(torch.tensor(_logit(initial_gate), dtype=torch.float))
        elif self.vee_type == "attention":
            self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.gamma_logit = nn.Parameter(torch.tensor(_logit(initial_gate), dtype=torch.float))
        elif self.vee_type == "gated_attention":
            self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            gate_hidden = min(128, self.hidden_dim)
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * self.hidden_dim, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 1),
            )
            nn.init.constant_(self.gate_mlp[-1].bias, _logit(initial_gate))

    def added_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _attention_evidence(
        self,
        hidden: torch.Tensor,
        visual: torch.Tensor,
        visual_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(hidden)
        k = self.k_proj(visual)
        u = self.v_proj(visual)
        scale = math.sqrt(self.hidden_dim) * max(self.attention_temperature, 1e-6)
        scores = torch.matmul(q, k.transpose(-1, -2)) / scale
        attn = _masked_softmax(scores, visual_mask)
        evidence = torch.matmul(attn, u)
        return evidence, attn

    def forward(
        self,
        hidden: torch.Tensor,
        visual: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        disable_visual: bool = False,
    ) -> VEEOutput:
        if self.vee_type == "none" or disable_visual:
            delta = torch.zeros_like(hidden)
            gate = hidden.new_zeros(hidden.shape[:-1] + (1,))
            return VEEOutput(hidden=hidden, delta=delta, gate=gate, evidence=delta, attention=None)

        if self.vee_type == "mean":
            pooled = _masked_mean(visual, visual_mask)
            evidence = self.evidence_proj(pooled).unsqueeze(1).expand_as(hidden)
            gate = torch.sigmoid(self.alpha_logit).to(dtype=hidden.dtype, device=hidden.device)
            delta = self.residual_scale * gate * evidence
            return VEEOutput(
                hidden=hidden + delta,
                delta=delta,
                gate=gate.expand(hidden.shape[:-1] + (1,)),
                evidence=evidence,
                attention=None,
            )

        evidence, attn = self._attention_evidence(hidden, visual, visual_mask)
        if self.vee_type == "attention":
            gate = torch.sigmoid(self.gamma_logit).to(dtype=hidden.dtype, device=hidden.device)
            delta = self.residual_scale * gate * evidence
            gate_tensor = gate.expand(hidden.shape[:-1] + (1,))
        else:
            gate_tensor = torch.sigmoid(self.gate_mlp(torch.cat([hidden, evidence], dim=-1)))
            delta = self.residual_scale * gate_tensor * evidence

        return VEEOutput(
            hidden=hidden + delta,
            delta=delta,
            gate=gate_tensor,
            evidence=evidence,
            attention=attn,
        )

    @torch.no_grad()
    def summarize_last(
        self,
        output: VEEOutput,
        span_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        if span_mask is None:
            mask = torch.ones(output.delta.shape[:-1], dtype=torch.bool, device=output.delta.device)
        else:
            mask = span_mask.to(dtype=torch.bool, device=output.delta.device)
        if mask.any():
            delta_norm = output.delta.norm(dim=-1)[mask]
            evidence_norm = output.evidence.norm(dim=-1)[mask]
            gate = output.gate.squeeze(-1)[mask]
            summary = {
                "delta_norm_mean": float(delta_norm.mean().detach().cpu()),
                "evidence_norm_mean": float(evidence_norm.mean().detach().cpu()),
                "gate_mean": float(gate.mean().detach().cpu()),
                "gate_std": float(gate.std(unbiased=False).detach().cpu()),
            }
        else:
            summary = {
                "delta_norm_mean": 0.0,
                "evidence_norm_mean": 0.0,
                "gate_mean": 0.0,
                "gate_std": 0.0,
            }
        if output.attention is not None:
            attn = output.attention.clamp_min(1e-12)
            ent = -(attn * attn.log()).sum(dim=-1)
            if mask.any():
                ent = ent[mask]
            summary["attention_entropy_mean"] = float(ent.mean().detach().cpu())
            summary["attention_max_mean"] = float(output.attention.max(dim=-1).values[mask].mean().detach().cpu())
        return summary

