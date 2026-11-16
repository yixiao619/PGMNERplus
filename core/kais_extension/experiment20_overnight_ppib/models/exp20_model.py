"""Experiment 20 models. Original PGMNER source is not modified."""
from __future__ import annotations

import torch
from torch import nn
from einops import rearrange, repeat

from models.bn_bert_ner_OT_robert import BertSpanNerBN as OriginalModel


class SampleResidual(nn.Module):
    def __init__(self, latent_dim=768, hidden_dim=1024, bottleneck=128, alpha=0.1):
        super().__init__()
        self.proj_in = nn.Linear(latent_dim * 2, bottleneck)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(bottleneck, hidden_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, zt, zv, spans):
        evidence = self.proj_out(self.act(self.proj_in(torch.cat([zt, zv], -1))))
        delta = self.alpha * evidence.unsqueeze(1).expand_as(spans)
        return delta, None


class EntityConditionedResidual(nn.Module):
    def __init__(self, latent_dim=768, hidden_dim=1024, bottleneck=128, alpha=0.5,
                 gate_mode="learned"):
        super().__init__()
        self.gate_mode = gate_mode
        self.evidence_in = nn.Linear(latent_dim * 2, bottleneck)
        self.evidence_out = nn.Linear(bottleneck, hidden_dim)
        self.gate_in = nn.Linear(hidden_dim + latent_dim * 2, bottleneck)
        self.gate_out = nn.Linear(bottleneck, 1)
        self.act = nn.GELU()
        nn.init.zeros_(self.evidence_out.weight)
        nn.init.zeros_(self.evidence_out.bias)
        nn.init.zeros_(self.gate_out.bias)
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, zt, zv, spans):
        global_z = torch.cat([zt, zv], -1)
        evidence = self.evidence_out(self.act(self.evidence_in(global_z))).unsqueeze(1)
        if self.gate_mode == "learned":
            expanded = global_z.unsqueeze(1).expand(-1, spans.size(1), -1)
            gamma = torch.sigmoid(self.gate_out(self.act(self.gate_in(
                torch.cat([spans, expanded], -1)))))
        elif self.gate_mode == "fixed_0.5":
            gamma = torch.full((*spans.shape[:2], 1), 0.5, device=spans.device,
                               dtype=spans.dtype)
        else:
            raise ValueError(self.gate_mode)
        return self.alpha * gamma * evidence, gamma.squeeze(-1)


class BertSpanNerBNExp20(OriginalModel):
    def __init__(self, config, args=None, num_labels=None):
        super().__init__(config, args=args, num_labels=num_labels)
        self.exp20_kind = str(args.exp20_kind)
        self.exp20_gate_mode = str(args.exp20_gate_mode)
        alpha = float(args.exp20_alpha)
        bottleneck = int(args.exp20_bottleneck)
        if self.exp20_kind == "sample":
            self.exp20_adapter = SampleResidual(768, int(args.hidden_dim), bottleneck, alpha)
        elif self.exp20_kind == "entity":
            self.exp20_adapter = EntityConditionedResidual(
                768, int(args.hidden_dim), bottleneck, alpha, self.exp20_gate_mode)
        else:
            raise ValueError(self.exp20_kind)
        self.last_exp20_summary = {}
        self.last_exp20_gate = None

    def _sample_latent(self, z, branch):
        if self.exp20_gate_mode == "learned":
            return z
        if self.exp20_gate_mode == "fixed_0.5":
            return 0.5 * z / branch.last_probe["gate"].clamp_min(1e-8)
        raise ValueError(self.exp20_gate_mode)

    def task_gate_parameters(self):
        if self.exp20_kind == "entity":
            return list(self.exp20_adapter.gate_in.parameters()) + list(
                self.exp20_adapter.gate_out.parameters())
        return list(self.dist_t.polar_gate.parameters()) + list(self.dist_v.polar_gate.parameters())

    def forward(self, ori_feas, only_test=False):
        if only_test:
            self.span_indexs.append(ori_feas["span_token_idxes"].cpu().detach())
            self.span_labels.append(ori_feas["span_labels"].cpu().detach())
            self.span_mask.append(ori_feas["span_masks"].cpu().detach())
        enc = {}
        enc["spans_rep"], enc["txt_rep_g"], attn_last = self.span_encoding(**ori_feas)
        text_local = enc["spans_rep"]
        if only_test:
            self.txt_representations_ori.append(text_local.cpu().detach())
        if self.use_pp_ib:
            mu_t, lv_t, t, polar_t = self.dist_t(enc["txt_rep_g"])
        else:
            mu_t, lv_t = self.dist_t(enc["txt_rep_g"])
            t, polar_t = self.reparameterise(mu_t, lv_t), None
        enc["imag_rep_g"], enc["imag_rep_l"] = self.image_model(ori_feas["image"])
        if self.use_pp_ib:
            mu_v, lv_v, v, polar_v = self.dist_v(enc["imag_rep_g"])
        else:
            mu_v, lv_v = self.dist_v(enc["imag_rep_g"])
            v, polar_v = self.reparameterise(mu_v, lv_v), None
        if only_test:
            self.txt_representations_IB.append(t.cpu().detach())
            self.img_representations_ori.append(enc["imag_rep_l"].cpu().detach())
            self.img_representations_IB.append(v.cpu().detach())
        loss_ot = ot_components = got_loss_cac = None
        if self.use_got or self.use_semantic_ot:
            tp, vp, at, av, plan, loss_ot, ot_components = self.semantic_ot_aligner(
                text_local, enc["imag_rep_l"], mask_t=ori_feas["span_masks"].float(),
                mask_v=None, attn_t=attn_last if self.use_got else None,
                span_token_idxes=ori_feas.get("span_token_idxes"), patch_pos=None)
        else:
            tp, vp, at, av, plan = self.optimal_transport_fusion(text_local, enc["imag_rep_l"])
        self.last_transport_plan = plan.detach().cpu()
        original = self.ln_text(self.fusion_text(torch.cat([tp, at], -1)) + at)
        zt = self._sample_latent(t, self.dist_t) if self.exp20_kind == "sample" else t
        zv = self._sample_latent(v, self.dist_v) if self.exp20_kind == "sample" else v
        delta, entity_gate = self.exp20_adapter(zt, zv, original)
        fused = original + delta
        mask = ori_feas["span_masks"].float()
        self.last_exp20_gate = entity_gate
        with torch.no_grad():
            valid = mask.sum().clamp_min(1)
            original_norm = (original.norm(dim=-1) * mask).sum() / valid
            residual_norm = (delta.norm(dim=-1) * mask).sum() / valid
            self.last_exp20_summary = {
                "residual_norm": float(residual_norm),
                "original_norm": float(original_norm),
                "residual_ratio": float(residual_norm / original_norm.clamp_min(1e-12)),
                "text_gate_mean": float(self.dist_t.last_probe["gate"].mean()),
                "visual_gate_mean": float(self.dist_v.last_probe["gate"].mean()),
            }
        logits = self.span_classifier(fused)
        bias = getattr(self, "_misc_logit_bias", None)
        if bias is not None and self.id_misc_focal >= 0:
            logits = logits.clone()
            logits[:, :, self.id_misc_focal] += float(bias)
        enc["logits"] = logits
        if only_test:
            self.fusion_text_representations.append(fused.cpu().detach())
        fusion_s = fused
        if len(fusion_s.size()) < 3:
            fusion_s = repeat(fusion_s, "b e -> r b e", r=1)
        fusion_s = repeat(rearrange(fusion_s, "r b e -> (r b) e"), "b e -> b v e", v=1)
        outputs = [self.softmax(logits)]
        loss = {}
        if ori_feas["span_labels"] is not None:
            loss = self.compute_loss(
                ori_feas, enc, t, v, mu_t, lv_t, mu_v, lv_v, fusion_s=fusion_s,
                transport_plan=plan, loss_ot=loss_ot, ot_components=ot_components,
                L_polar_t=polar_t, L_polar_v=polar_v, got_loss_cac=got_loss_cac,
                fused_text=fused, model_training=self.training)
        return outputs, loss
