# Round 2: span-grid auxiliary head + classifier bias init + IB latent pairing (no CRF in this codebase).
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpanEntityDetectionHead(nn.Module):
    """Binary entity vs O on each span cell (same grid as span classifier)."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, span_h: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(span_h))

    @staticmethod
    def compute_loss(
        logits: torch.Tensor,
        span_labels: torch.Tensor,
        span_masks: torch.Tensor,
        o_label_id: int = 0,
    ) -> torch.Tensor:
        """
        logits: (B, S, 2)
        span_labels: (B, S) long, 0 = O
        span_masks: (B, S) bool or float
        """
        m = span_masks.bool()
        tgt = (span_labels != o_label_id).long()
        logits_f = logits[m]
        tgt_f = tgt[m]
        if logits_f.numel() == 0:
            return logits.new_tensor(0.0)

        n_pos = tgt_f.sum().float()
        n_neg = (1 - tgt_f).sum().float()
        pos_w = torch.clamp(n_neg / (n_pos + 1e-8), max=10.0)
        w = logits.new_tensor([1.0, pos_w.item()])
        return F.cross_entropy(logits_f, tgt_f, weight=w)


def init_span_classifier_output_bias(
    classifier: nn.Module,
    label_names: tuple,
    misc_bias: float = 0.0,
    entity_bias: float = 0.0,
    o_penalty: float = 0.0,
) -> None:
    """
    MultiNonLinearClassifier: adjust classifier2.bias (analogous to CRF O→B-* priors).
    label_names order must match logits (e.g. O, LOC, MISC, ORG, PER).
    """
    if misc_bias == 0.0 and entity_bias == 0.0 and o_penalty == 0.0:
        return
    if not hasattr(classifier, "classifier2") or not hasattr(classifier.classifier2, "bias"):
        return
    bias = classifier.classifier2.bias
    with torch.no_grad():
        for i, name in enumerate(label_names):
            if name == "O":
                bias[i] += o_penalty
            elif name == "MISC":
                bias[i] += misc_bias
            else:
                bias[i] += entity_bias


def compute_ib_global_paired_contrastive_loss(
    text_latent: torch.Tensor,
    image_latent: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    PP-IB latents are global (B, D), not per-token. CLIP-style batch contrastive on pairs (t_i, v_i).
    """
    if text_latent.shape[0] < 2:
        return text_latent.new_tensor(0.0)
    t = F.normalize(text_latent, dim=-1)
    v = F.normalize(image_latent, dim=-1)
    sim = torch.matmul(t, v.T) / float(temperature)
    labels = torch.arange(t.size(0), device=t.device, dtype=torch.long)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) * 0.5
