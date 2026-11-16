"""PGMNER + VEE subclass for Experiment 5.

The original PGMNER implementation is not modified. This subclass preserves
the original PP-IB, GOT-A/Partial OT, losses, and classifier, and inserts VEE
as a residual path immediately before the original span classifier.
"""

from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange, repeat

from models.bn_bert_ner_OT_robert import BertSpanNerBN as OriginalBertSpanNerBN
from .visual_evidence_enhancer import VisualEvidenceEnhancer


class BertSpanNerBNVEE(OriginalBertSpanNerBN):
    def __init__(self, config, args=None, num_labels=None):
        super().__init__(config, args=args, num_labels=num_labels)
        self.vee_type = str(getattr(args, "vee_type", "none") or "none").lower()
        self.vee = VisualEvidenceEnhancer(
            hidden_dim=int(args.hidden_dim),
            vee_type=self.vee_type,
            initial_gate=float(getattr(args, "vee_initial_gate", 0.1)),
            residual_scale=float(getattr(args, "vee_residual_scale", 1.0)),
            attention_temperature=float(getattr(args, "vee_attention_temperature", 1.0)),
        )
        self.disable_vee_visual = bool(getattr(args, "disable_vee_visual", False))
        self.last_vee_summary = {}

    def set_vee_visual_enabled(self, enabled: bool) -> None:
        self.disable_vee_visual = not bool(enabled)

    def forward(self, ori_feas, only_test=False):
        if only_test:
            self.span_indexs.append(ori_feas["span_token_idxes"].cpu().detach())
            self.span_labels.append(ori_feas["span_labels"].cpu().detach())
            self.span_mask.append(ori_feas["span_masks"].cpu().detach())

        ori_encoding = {}
        ori_encoding["spans_rep"], ori_encoding["txt_rep_g"], attn_last = self.span_encoding(**ori_feas)
        text_embed_local = ori_encoding["spans_rep"]
        if only_test:
            self.txt_representations_ori.append(ori_encoding["spans_rep"].cpu().detach())

        if self.use_pp_ib:
            mu_t_dist, logvar_t_dist, t, L_polar_t = self.dist_t(ori_encoding["txt_rep_g"])
        else:
            mu_t_dist, logvar_t_dist = self.dist_t(ori_encoding["txt_rep_g"])
            t = self.reparameterise(mu_t_dist, logvar_t_dist)
            L_polar_t = None
        if only_test:
            self.txt_representations_IB.append(t.cpu().detach())

        images = ori_feas["image"]
        ori_encoding["imag_rep_g"], ori_encoding["imag_rep_l"] = self.image_model(images)
        if only_test:
            self.img_representations_ori.append(ori_encoding["imag_rep_l"].cpu().detach())

        if self.use_pp_ib:
            mu_v_dist, logvar_v_dist, v, L_polar_v = self.dist_v(ori_encoding["imag_rep_g"])
        else:
            mu_v_dist, logvar_v_dist = self.dist_v(ori_encoding["imag_rep_g"])
            v = self.reparameterise(mu_v_dist, logvar_v_dist)
            L_polar_v = None
        if only_test:
            self.img_representations_IB.append(v.cpu().detach())

        vis_embed_local = ori_encoding["imag_rep_l"]

        loss_ot = None
        ot_components = None
        got_loss_cac = None
        if getattr(self.args, "desc_fusion", "concat") == "separate":
            raise NotImplementedError("desc_fusion=separate is not implemented; use concat (PGIM-style).")
        if getattr(self.args, "use_vllm_desc", False):
            from utils.vllm_sanity import assert_vllm_desc_batch

            assert_vllm_desc_batch(ori_feas)

        if self.use_got or self.use_semantic_ot:
            mask_t = ori_feas["span_masks"].float()
            attn_t = attn_last if self.use_got else None
            (
                text_proj,
                visual_proj,
                align_text,
                align_visual,
                transport_plan,
                loss_ot,
                ot_components,
            ) = self.semantic_ot_aligner(
                text_embed_local,
                vis_embed_local,
                mask_t=mask_t,
                mask_v=None,
                attn_t=attn_t,
                span_token_idxes=ori_feas.get("span_token_idxes"),
                patch_pos=None,
            )
        else:
            (
                text_proj,
                visual_proj,
                align_text,
                align_visual,
                transport_plan,
            ) = self.optimal_transport_fusion(text_embed_local, vis_embed_local)
            got_loss_cac = None

        self.last_transport_plan = transport_plan.detach().cpu()

        if only_test:
            self.transport_plans.append(transport_plan.cpu().detach())
            self.txt_representations_OT_g.append(text_proj.cpu().detach())
            self.txt_representations_OT.append(ori_encoding["txt_rep_g"].cpu().detach())
            self.img_representations_OT_g.append(visual_proj.cpu().detach())
            self.img_representations_OT.append(ori_encoding["imag_rep_g"].cpu().detach())

        text_combined = torch.cat([text_proj, align_text], dim=-1)
        fused_text = self.fusion_text(text_combined)
        fused_text = self.ln_text(fused_text + align_text)

        vee_out = self.vee(
            fused_text,
            visual_proj,
            visual_mask=None,
            disable_visual=self.disable_vee_visual,
        )
        fused_text = vee_out.hidden
        self.last_vee_summary = self.vee.summarize_last(
            vee_out,
            span_mask=ori_feas.get("span_masks"),
        )

        logits = self.span_classifier(fused_text)
        bias = getattr(self, "_misc_logit_bias", None)
        if bias is not None and self.id_misc_focal >= 0:
            logits = logits.clone()
            logits[:, :, self.id_misc_focal] = logits[:, :, self.id_misc_focal] + float(bias)
        ori_encoding["logits"] = logits
        if only_test:
            self.fusion_text_representations.append(fused_text.cpu().detach())

        fusion_s = fused_text
        if len(fusion_s.size()) < 3:
            fusion_s = repeat(fusion_s, "b e -> r b e", r=1)
        fusion_s = repeat(rearrange(fusion_s, "r b e -> (r b) e"), "b e ->b v e", v=1)

        loss_dict = {}
        outputs = [self.softmax(logits)]

        if ori_feas["span_labels"] is not None:
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

