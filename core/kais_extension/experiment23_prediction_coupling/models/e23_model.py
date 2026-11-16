"""Prediction-coupled GOT-A variants for Experiment 23 only."""
from __future__ import annotations
import torch
from torch import nn
from models.bn_bert_ner_OT_robert import BertSpanNerBN as OriginalModel


class NormalizedOTCoupler(nn.Module):
    """Normalize each transport row before a small residual injection."""
    def __init__(self, hidden_dim: int, route: str):
        super().__init__()
        self.route = route
        self.evidence_norm = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        nn.init.eye_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        if route == "entity_routing":
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 2, 64), nn.GELU(), nn.Linear(64, 1)
            )
            nn.init.normal_(self.gate[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.gate[-1].bias)
            self.scale = 0.10  # effective initial scale is approximately 0.05
        elif route == "ot_fusion":
            self.gate = None
            self.scale = 0.05
        else:
            raise ValueError(route)

    def forward(self, fused, visual_proj, plan, span_mask, shuffle_plan=False):
        if shuffle_plan:
            plan = plan.roll(shifts=1, dims=-1)  # T changes while V does not.
        row_mass = plan.sum(-1, keepdim=True)
        # The learned plan is nearly diffuse. Restrict aggregation to its
        # highest-mass patches so the residual depends on correspondence,
        # rather than degenerating to the global visual mean.
        k = min(16, plan.size(-1))
        _, top_idx = plan.topk(k=k, dim=-1)
        sparse_plan = torch.zeros_like(plan).scatter_(-1, top_idx, 1.0) * plan
        sparse_mass = sparse_plan.sum(-1, keepdim=True)
        evidence = torch.bmm(sparse_plan, visual_proj) / sparse_mass.clamp_min(1e-8)
        evidence = self.evidence_norm(evidence)
        projected = self.projection(evidence)
        concentration = plan.amax(-1, keepdim=True) / row_mass.clamp_min(1e-8)
        valid_count = span_mask.sum(1, keepdim=True).clamp_min(1.0)
        expected_mass = plan.sum((1, 2), keepdim=True) / valid_count.unsqueeze(-1)
        relative_mass = (row_mass / expected_mass.clamp_min(1e-8)).clamp(0.0, 2.0)
        if self.gate is None:
            gamma = torch.ones_like(row_mass)
        else:
            gate_input = torch.cat(
                [torch.nn.functional.layer_norm(fused, (fused.size(-1),)),
                 evidence, relative_mass, concentration], dim=-1)
            gamma = torch.sigmoid(self.gate(gate_input))
        delta = self.scale * gamma * projected * span_mask.unsqueeze(-1)
        return fused + delta, delta, gamma, row_mass, concentration


class PredictionCoupledPGMNER(OriginalModel):
    """Original forward with hook-based, experiment-local classifier coupling."""
    def __init__(self, config, args=None, num_labels=None):
        super().__init__(config, args=args, num_labels=num_labels)
        self.e23_route = str(args.e23_route)
        self.e23_coupler = NormalizedOTCoupler(int(args.hidden_dim), self.e23_route)
        self.e23_enabled = True
        self.e23_shuffle_plan = False
        self.last_e23_summary = {}
        self.last_e23_gate = None
        self._e23_alignment = None
        self._e23_span_mask = None

    def coupling_parameters(self):
        return list(self.e23_coupler.parameters())

    def _capture_alignment(self, module, inputs, output):
        del module, inputs
        self._e23_alignment = output

    def _inject_before_classifier(self, module, inputs):
        del module
        fused = inputs[0]
        if not self.e23_enabled or self._e23_alignment is None:
            self.last_e23_summary = {"residual_ratio": 0.0}
            self.last_e23_gate = None
            return inputs
        _, visual_proj, _, _, plan, _, _ = self._e23_alignment
        out, delta, gamma, row_mass, concentration = self.e23_coupler(
            fused, visual_proj, plan, self._e23_span_mask,
            shuffle_plan=self.e23_shuffle_plan)
        mask = self._e23_span_mask
        with torch.no_grad():
            den = mask.sum().clamp_min(1.0)
            base_norm = (fused.norm(dim=-1) * mask).sum() / den
            delta_norm = (delta.norm(dim=-1) * mask).sum() / den
            valid_gate = gamma.squeeze(-1)[mask.bool()]
            self.last_e23_summary = {
                "residual_ratio": float(delta_norm / base_norm.clamp_min(1e-12)),
                "residual_norm": float(delta_norm),
                "gate_mean": float(valid_gate.mean()),
                "gate_std": float(valid_gate.std(unbiased=False)),
                "span_gate_variance": float(valid_gate.var(unbiased=False)),
                "row_mass_mean": float((row_mass.squeeze(-1) * mask).sum() / den),
                "concentration_mean": float((concentration.squeeze(-1) * mask).sum() / den),
            }
            self.last_e23_gate = gamma.detach()
        return (out,) + tuple(inputs[1:])

    def forward(self, ori_feas, only_test=False):
        self._e23_span_mask = ori_feas["span_masks"].float()
        h_align = self.semantic_ot_aligner.register_forward_hook(self._capture_alignment)
        h_class = self.span_classifier.register_forward_pre_hook(
            self._inject_before_classifier)
        try:
            return super().forward(ori_feas, only_test=only_test)
        finally:
            h_class.remove(); h_align.remove(); self._e23_alignment = None


__all__ = ["PredictionCoupledPGMNER"]
