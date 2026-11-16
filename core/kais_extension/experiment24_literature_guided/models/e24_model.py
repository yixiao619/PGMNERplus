"""Experiment-local task-guided OT supervision; original PGMNER files stay unchanged."""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from models.bn_bert_ner_OT_robert import BertSpanNerBN as OriginalModel


class TaskGuidedOTPGMNER(OriginalModel):
    """Add label supervision to transport-derived evidence and plan polarization."""

    def __init__(self, config, args=None, num_labels=None):
        super().__init__(config, args=args, num_labels=num_labels)
        dim = int(args.hidden_dim)
        self.e24_visual_classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.n_class))
        self.e24_aux_weight = float(getattr(args, "e24_aux_weight", 0.10))
        self.e24_polar_weight = float(getattr(args, "e24_polar_weight", 0.01))
        self._e24_alignment = None
        self.last_e24_stats = {}

    def _capture(self, module, inputs, output):
        del module, inputs
        self._e24_alignment = output

    def route_parameters(self):
        return list(self.e24_visual_classifier.parameters())

    def forward(self, ori_feas, only_test=False):
        hook = self.semantic_ot_aligner.register_forward_hook(self._capture)
        try:
            outputs, losses = super().forward(ori_feas, only_test=only_test)
        finally:
            hook.remove()
        if self._e24_alignment is None or ori_feas.get("span_labels") is None:
            return outputs, losses
        _, visual, _, _, plan, _, _ = self._e24_alignment
        span_mask = ori_feas["span_masks"].bool()
        labels = ori_feas["span_labels"]
        entity_mask = span_mask & labels.ne(0)
        row_mass = plan.sum(-1, keepdim=True)
        row_prob = plan / row_mass.clamp_min(1e-12)
        evidence = torch.bmm(row_prob, visual)
        aux_logits = self.e24_visual_classifier(evidence)
        if entity_mask.any():
            aux = F.cross_entropy(aux_logits[entity_mask], labels[entity_mask])
            nv = plan.size(-1)
            lo, hi = 0.5 / float(nv), 2.0 / float(nv)
            polar = F.relu(-((row_prob - lo) * (row_prob - hi)))[entity_mask].mean()
            extra = self.e24_aux_weight * aux + self.e24_polar_weight * polar
        else:
            aux = polar = plan.sum() * 0.0
            extra = aux
        losses["e24_entity_visual_ce"] = aux
        losses["e24_plan_polar"] = polar
        losses["e24_extra"] = extra
        losses["loss"] = losses["loss"] + extra
        with torch.no_grad():
            p = row_prob[entity_mask] if entity_mask.any() else row_prob[span_mask]
            ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1) / math.log(max(2, p.size(-1)))
            self.last_e24_stats = {
                "transport_entropy": float(ent.mean()) if ent.numel() else 0.0,
                "transport_max": float(p.max(-1).values.mean()) if p.numel() else 0.0,
                "transport_row_mass": float(row_mass.squeeze(-1)[span_mask].mean()),
                "entity_spans": int(entity_mask.sum()),
            }
        return outputs, losses


__all__ = ["TaskGuidedOTPGMNER"]
