"""Prediction-coupled PP-IB model for Experiment 19 only.

The original model is subclassed, not modified. Its sample-level gated text
and visual latents are projected to one residual vector and broadcast over all
candidate spans immediately before the unchanged span classifier.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from einops import rearrange, repeat
from models.bn_bert_ner_OT_robert import BertSpanNerBN as OriginalBertSpanNerBN

class PCPPIBAdapter(nn.Module):
    def __init__(self, latent_dim=768, hidden_dim=1024, bottleneck=128, alpha=0.1):
        super().__init__()
        self.proj_in=nn.Linear(latent_dim*2,bottleneck)
        self.act=nn.GELU()
        self.proj_out=nn.Linear(bottleneck,hidden_dim)
        nn.init.zeros_(self.proj_out.weight); nn.init.zeros_(self.proj_out.bias)
        self.register_buffer('alpha',torch.tensor(float(alpha)))
    def forward(self,zt,zv): return self.alpha*self.proj_out(self.act(self.proj_in(torch.cat([zt,zv],-1))))

class BertSpanNerBNPCPPIB(OriginalBertSpanNerBN):
    def __init__(self,config,args=None,num_labels=None):
        super().__init__(config,args=args,num_labels=num_labels)
        self.pc_gate_mode=str(getattr(args,'pc_gate_mode','learned'))
        self.pc_adapter=PCPPIBAdapter(768,int(args.hidden_dim),int(getattr(args,'pc_bottleneck',128)),float(getattr(args,'pc_alpha',.1)))
        self.last_pc_summary={}
    def _pc_latent(self,z,branch):
        if self.pc_gate_mode=='learned': return z
        if self.pc_gate_mode=='fixed_0.5': return .5*z/branch.last_probe['gate'].clamp_min(1e-8)
        raise ValueError(self.pc_gate_mode)
    def forward(self,ori_feas,only_test=False):
        if only_test:
            self.span_indexs.append(ori_feas['span_token_idxes'].cpu().detach());self.span_labels.append(ori_feas['span_labels'].cpu().detach());self.span_mask.append(ori_feas['span_masks'].cpu().detach())
        enc={};enc['spans_rep'],enc['txt_rep_g'],attn_last=self.span_encoding(**ori_feas);text_local=enc['spans_rep']
        if only_test:self.txt_representations_ori.append(text_local.cpu().detach())
        if self.use_pp_ib: mu_t,lv_t,t,Lpt=self.dist_t(enc['txt_rep_g'])
        else: mu_t,lv_t=self.dist_t(enc['txt_rep_g']);t=self.reparameterise(mu_t,lv_t);Lpt=None
        images=ori_feas['image'];enc['imag_rep_g'],enc['imag_rep_l']=self.image_model(images)
        if self.use_pp_ib: mu_v,lv_v,v,Lpv=self.dist_v(enc['imag_rep_g'])
        else: mu_v,lv_v=self.dist_v(enc['imag_rep_g']);v=self.reparameterise(mu_v,lv_v);Lpv=None
        if only_test:
            self.txt_representations_IB.append(t.cpu().detach());self.img_representations_ori.append(enc['imag_rep_l'].cpu().detach());self.img_representations_IB.append(v.cpu().detach())
        loss_ot=otc=got_cac=None
        if self.use_got or self.use_semantic_ot:
            tp,vp,at,av,T,loss_ot,otc=self.semantic_ot_aligner(text_local,enc['imag_rep_l'],mask_t=ori_feas['span_masks'].float(),mask_v=None,attn_t=attn_last if self.use_got else None,span_token_idxes=ori_feas.get('span_token_idxes'),patch_pos=None)
        else:
            tp,vp,at,av,T=self.optimal_transport_fusion(text_local,enc['imag_rep_l'])
        self.last_transport_plan=T.detach().cpu()
        original=self.ln_text(self.fusion_text(torch.cat([tp,at],-1))+at)
        zt=self._pc_latent(t,self.dist_t);zv=self._pc_latent(v,self.dist_v);delta=self.pc_adapter(zt,zv);fused=original+delta.unsqueeze(1)
        with torch.no_grad():
            mask=ori_feas['span_masks'].float(); orig_norm=(original.norm(dim=-1)*mask).sum()/mask.sum().clamp_min(1); ratio=delta.norm(dim=-1).mean()/orig_norm.clamp_min(1e-12)
            self.last_pc_summary={'residual_norm':float(delta.norm(dim=-1).mean()),'original_norm':float(orig_norm),'residual_ratio':float(ratio),'text_gate_mean':float(self.dist_t.last_probe['gate'].mean()),'text_gate_std':float(self.dist_t.last_probe['gate'].std(unbiased=False)),'visual_gate_mean':float(self.dist_v.last_probe['gate'].mean()),'visual_gate_std':float(self.dist_v.last_probe['gate'].std(unbiased=False))}
        logits=self.span_classifier(fused);bias=getattr(self,'_misc_logit_bias',None)
        if bias is not None and self.id_misc_focal>=0:
            logits=logits.clone();logits[:,:,self.id_misc_focal]+=float(bias)
        enc['logits']=logits
        if only_test:self.fusion_text_representations.append(fused.cpu().detach())
        fs=fused
        if len(fs.size())<3:fs=repeat(fs,'b e -> r b e',r=1)
        fs=repeat(rearrange(fs,'r b e -> (r b) e'),'b e ->b v e',v=1)
        outputs=[self.softmax(logits)];loss={}
        if ori_feas['span_labels'] is not None:
            loss=self.compute_loss(ori_feas,enc,t,v,mu_t,lv_t,mu_v,lv_v,fusion_s=fs,transport_plan=T,loss_ot=loss_ot,ot_components=otc,L_polar_t=Lpt,L_polar_v=Lpv,got_loss_cac=got_cac,fused_text=fused,model_training=self.training)
        return outputs,loss
