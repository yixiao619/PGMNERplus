from __future__ import annotations
import math
import torch
from torch import nn
from models.partial_ot import PartialSinkhorn, NodeSinkhorn

def _sinkhorn(C,a,b,eps,iters):
    la=torch.log(a.clamp_min(1e-12));lb=torch.log(b.clamp_min(1e-12));K=-C/eps;u=torch.zeros_like(la);v=torch.zeros_like(lb)
    for _ in range(iters):
        v=lb-torch.logsumexp(K+u.unsqueeze(2),dim=1)
        u=la-torch.logsumexp(K+v.unsqueeze(1),dim=2)
    return torch.exp(K+u.unsqueeze(2)+v.unsqueeze(1))

class PerSamplePartialSinkhorn(PartialSinkhorn):
    """Partial Sinkhorn supporting an actual mass for each sample."""
    def actual_mass(self,z_text,z_vis,C,mt,mv):
        return self.get_s().to(C).expand(C.size(0))
    def adjusted_cost(self,z_text,z_vis,C,mt,mv): return C
    def forward(self,z_text,z_vis,C_pre=None,mask_t=None,mask_v=None,invalid_cost=1e4):
        B,n,_=z_text.shape;m=z_vis.size(1);C=C_pre if C_pre is not None else self._cosine_cost(z_text,z_vis);dev=C.device;dt=C.dtype
        mt=torch.ones(B,n,device=dev,dtype=dt) if mask_t is None else mask_t.to(dev,dtype=dt).float();mv=torch.ones(B,m,device=dev,dtype=dt) if mask_v is None else mask_v.to(dev,dtype=dt).float();C=self.adjusted_cost(z_text,z_vis,C,mt,mv)
        s=self.actual_mass(z_text,z_vis,C,mt,mv).clamp(.3,.9);work=C+(1-mt[:,:,None]*mv[:,None,:])*invalid_cost
        aug=torch.full((B,n+1,m+1),self.dust_cost,device=dev,dtype=dt);aug[:,:n,:m]=work;aug[:,n,m]=0
        a=torch.zeros(B,n+1,device=dev,dtype=dt);b=torch.zeros(B,m+1,device=dev,dtype=dt);a[:,:n]=s[:,None]*mt/mt.sum(1,keepdim=True).clamp_min(1e-12);a[:,n]=1-s;b[:,:m]=s[:,None]*mv/mv.sum(1,keepdim=True).clamp_min(1e-12);b[:,m]=1-s
        A=_sinkhorn(aug,a,b,self.eps,self.n_iters);T=A[:,:n,:m];self.last_factory_stats={"mass_mean":float(s.detach().mean()),"mass_std":float(s.detach().std()),"reliability_std":0.0};return T,(T*C*mt[:,:,None]*mv[:,None,:]).sum((1,2)).mean()

class AdaptiveMassSinkhorn(PerSamplePartialSinkhorn):
    def __init__(self,*a,mass_min=.3,mass_max=.9,initial_mass=.594940338282,**kw):
        super().__init__(*a,**kw);self.mass_min=float(mass_min);self.mass_max=float(mass_max);self.mass_net=nn.Sequential(nn.Linear(3,16),nn.GELU(),nn.Linear(16,1));nn.init.zeros_(self.mass_net[-1].weight);p=(initial_mass-self.mass_min)/(self.mass_max-self.mass_min);nn.init.constant_(self.mass_net[-1].bias,math.log(p/(1-p)))
    def actual_mass(self,z_text,z_vis,C,mt,mv):
        mask=mt[:,:,None]*mv[:,None,:];den=mask.sum((1,2)).clamp_min(1);mean=(C*mask).sum((1,2))/den;var=(((C-mean[:,None,None])**2)*mask).sum((1,2))/den;minimum=(C+(1-mask)*1e4).amin((1,2));x=torch.stack([mean,var.sqrt(),minimum],1);s=self.mass_min+(self.mass_max-self.mass_min)*torch.sigmoid(self.mass_net(x).squeeze(-1));return s

class FixedActualMassSinkhorn(PerSamplePartialSinkhorn):
    def __init__(self,*a,actual_mass=.594940338282,**kw):super().__init__(*a,**kw);self.fixed_actual_mass=float(actual_mass)
    def actual_mass(self,z_text,z_vis,C,mt,mv):return C.new_full((C.size(0),),self.fixed_actual_mass)

class ReliabilityWeightedSinkhorn(PerSamplePartialSinkhorn):
    def __init__(self,*a,dim=1024,strength=1.0,initial_mass=.594940338282,**kw):
        super().__init__(*a,**kw);p=(float(initial_mass)-.3)/.6;self.s_logit.data.fill_(math.log(p/(1-p)));self.text_reliability=nn.Linear(dim,1);self.visual_reliability=nn.Linear(dim,1);nn.init.zeros_(self.text_reliability.weight);nn.init.zeros_(self.text_reliability.bias);nn.init.zeros_(self.visual_reliability.weight);nn.init.zeros_(self.visual_reliability.bias);self.strength=float(strength)
    def adjusted_cost(self,z_text,z_vis,C,mt,mv):
        rt=2*torch.sigmoid(self.text_reliability(z_text)).squeeze(-1);rv=2*torch.sigmoid(self.visual_reliability(z_vis)).squeeze(-1);factor=(rt[:,:,None]*rv[:,None,:]).clamp_min(.1).sqrt();out=C/((1-self.strength)+self.strength*factor);self.last_reliability=(rt.detach(),rv.detach());return out
    def forward(self,*a,**kw):
        T,L=super().forward(*a,**kw)
        if hasattr(self,"last_reliability"):
            rt,rv=self.last_reliability;self.last_factory_stats["reliability_std"]=float(torch.cat([rt.flatten(),rv.flatten()]).std())
        return T,L

__all__=["NodeSinkhorn","AdaptiveMassSinkhorn","FixedActualMassSinkhorn","ReliabilityWeightedSinkhorn"]
