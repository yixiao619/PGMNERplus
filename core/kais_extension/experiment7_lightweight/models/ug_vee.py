"""Uncertainty-gated entity-aware VEE for Experiment 7B."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from einops import rearrange, repeat

from models.bn_bert_ner_OT_robert import BertSpanNerBN as OriginalBertSpanNerBN


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class UncertaintyGatedVEE(nn.Module):
    def __init__(self, hidden_dim: int, initial_alpha: float = 0.1, temperature: float = 1.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.alpha_logit = nn.Parameter(torch.tensor(_logit(initial_alpha), dtype=torch.float))
        self.last_summary = {}

    def forward(self, hidden, visual, baseline_logits, active_margin_threshold: float, disable_visual: bool = False):
        probs = torch.softmax(baseline_logits.detach(), dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1).values
        margin = (top2[..., 0] - top2[..., 1]).detach()
        if disable_visual:
            gate = torch.zeros_like(margin).unsqueeze(-1)
            delta = torch.zeros_like(hidden)
            self.last_summary = {
                "alpha": float(torch.sigmoid(self.alpha_logit.detach()).cpu()),
                "active_fraction": 0.0,
                "gate_mean": 0.0,
                "delta_norm_mean": 0.0,
                "hidden_norm_mean": float(hidden.detach().norm(dim=-1).mean().cpu()),
                "residual_ratio": 0.0,
                "attention_entropy": 0.0,
                "attention_max": 0.0,
            }
            return hidden, delta, gate
        active = (margin <= float(active_margin_threshold)).to(hidden.dtype)
        # uncertainty is largest when top1/top2 margin is small; keep stop-gradient.
        denom = max(float(active_margin_threshold), 1e-6)
        gate_scalar = active * torch.clamp((float(active_margin_threshold) - margin) / denom, 0.0, 1.0)
        q = self.q_proj(hidden)
        k = self.k_proj(visual)
        v = self.v_proj(visual)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (math.sqrt(self.hidden_dim) * max(self.temperature, 1e-6))
        attn = torch.softmax(scores, dim=-1)
        evidence = torch.matmul(attn, v)
        alpha = torch.sigmoid(self.alpha_logit).to(dtype=hidden.dtype, device=hidden.device)
        delta = alpha * gate_scalar.unsqueeze(-1) * evidence
        out = hidden + delta
        with torch.no_grad():
            hn = hidden.detach().norm(dim=-1).clamp_min(1e-12)
            dn = delta.detach().norm(dim=-1)
            ent = -(attn.detach().clamp_min(1e-12) * attn.detach().clamp_min(1e-12).log()).sum(dim=-1)
            self.last_summary = {
                "alpha": float(alpha.detach().cpu()),
                "active_fraction": float((active > 0).float().mean().detach().cpu()),
                "gate_mean": float(gate_scalar.mean().detach().cpu()),
                "delta_norm_mean": float(dn.mean().detach().cpu()),
                "hidden_norm_mean": float(hn.mean().detach().cpu()),
                "residual_ratio": float((dn / hn).mean().detach().cpu()),
                "attention_entropy": float(ent.mean().detach().cpu()),
                "attention_max": float(attn.detach().max(dim=-1).values.mean().cpu()),
            }
        return out, delta, gate_scalar.unsqueeze(-1)


class BertSpanNerBNUgVEE(OriginalBertSpanNerBN):
    def __init__(self, config, args=None, num_labels=None):
        super().__init__(config, args=args, num_labels=num_labels)
        self.ug_margin_threshold = float(getattr(args, "ug_margin_threshold", 0.5))
        self.disable_new_visual_path = bool(getattr(args, "disable_new_visual_path", False))
        self.ug_vee = UncertaintyGatedVEE(
            hidden_dim=int(args.hidden_dim),
            initial_alpha=float(getattr(args, "ug_initial_alpha", 0.1)),
            temperature=float(getattr(args, "ug_temperature", 1.0)),
        )
        self.last_ug_summary = {}

    def set_new_visual_enabled(self, enabled: bool):
        self.disable_new_visual_path = not bool(enabled)

    def forward(self, ori_feas, only_test=False):
        if only_test:
            self.span_indexs.append(ori_feas['span_token_idxes'].cpu().detach())
            self.span_labels.append(ori_feas['span_labels'].cpu().detach())
            self.span_mask.append(ori_feas['span_masks'].cpu().detach())
        ori_encoding = {}
        ori_encoding['spans_rep'], ori_encoding['txt_rep_g'], attn_last = self.span_encoding(**ori_feas)
        text_embed_local = ori_encoding['spans_rep']
        if self.use_pp_ib:
            mu_t_dist, logvar_t_dist, t, L_polar_t = self.dist_t(ori_encoding['txt_rep_g'])
        else:
            mu_t_dist, logvar_t_dist = self.dist_t(ori_encoding['txt_rep_g'])
            t = self.reparameterise(mu_t_dist, logvar_t_dist)
            L_polar_t = None
        images = ori_feas['image']
        ori_encoding['imag_rep_g'], ori_encoding['imag_rep_l'] = self.image_model(images)
        if self.use_pp_ib:
            mu_v_dist, logvar_v_dist, v, L_polar_v = self.dist_v(ori_encoding['imag_rep_g'])
        else:
            mu_v_dist, logvar_v_dist = self.dist_v(ori_encoding['imag_rep_g'])
            v = self.reparameterise(mu_v_dist, logvar_v_dist)
            L_polar_v = None
        vis_embed_local = ori_encoding['imag_rep_l']
        loss_ot = None
        ot_components = None
        got_loss_cac = None
        if self.use_got or self.use_semantic_ot:
            mask_t = ori_feas['span_masks'].float()
            attn_t = attn_last if self.use_got else None
            (text_proj, visual_proj, align_text, align_visual, transport_plan, loss_ot, ot_components) = self.semantic_ot_aligner(
                text_embed_local,
                vis_embed_local,
                mask_t=mask_t,
                mask_v=None,
                attn_t=attn_t,
                span_token_idxes=ori_feas.get("span_token_idxes"),
                patch_pos=None,
            )
        else:
            (text_proj, visual_proj, align_text, align_visual, transport_plan) = self.optimal_transport_fusion(text_embed_local, vis_embed_local)
        self.last_transport_plan = transport_plan.detach().cpu()
        text_combined = torch.cat([text_proj, align_text], dim=-1)
        fused_text = self.fusion_text(text_combined)
        fused_text = self.ln_text(fused_text + align_text)
        baseline_logits = self.span_classifier(fused_text)
        fused_text, _, _ = self.ug_vee(
            fused_text,
            visual_proj,
            baseline_logits,
            active_margin_threshold=self.ug_margin_threshold,
            disable_visual=self.disable_new_visual_path,
        )
        self.last_ug_summary = dict(self.ug_vee.last_summary)
        logits = self.span_classifier(fused_text)
        bias = getattr(self, "_misc_logit_bias", None)
        if bias is not None and self.id_misc_focal >= 0:
            logits = logits.clone()
            logits[:, :, self.id_misc_focal] = logits[:, :, self.id_misc_focal] + float(bias)
        ori_encoding['logits'] = logits
        fusion_s = fused_text
        if len(fusion_s.size()) < 3:
            fusion_s = repeat(fusion_s, 'b e -> r b e', r=1)
        fusion_s = repeat(rearrange(fusion_s, 'r b e -> (r b) e'), 'b e ->b v e', v=1)
        outputs = [self.softmax(logits)]
        loss_dict = {}
        if ori_feas['span_labels'] is not None:
            loss_dict = self.compute_loss(
                ori_feas,
                ori_encoding,
                t,
                v,
                mu_t_dist,
                logvar_t_dist,
                mu_v_dist,
                logvar_v_dist,
                fusion_s=fusion_s,
                transport_plan=transport_plan,
                loss_ot=loss_ot,
                ot_components=ot_components,
                L_polar_t=L_polar_t,
                L_polar_v=L_polar_v,
                got_loss_cac=got_loss_cac,
                fused_text=fused_text,
                model_training=self.training,
            )
        return outputs, loss_dict
