"""Opt-in experimental repair. No historical checkpoint compatibility claim."""
import math
import os
import json
import torch
from torch import nn
from torch.nn import functional as F


class LocalPolarizedIB(nn.Module):
    """Local span/patch Gaussian code; task-connected soft gate and prior noise.

    Evaluation uses E[g*z+(1-g)*noise]=g*mu for deterministic predictions.
    Span-level text granularity is retained; this is not a BIO/CRF reproduction.
    """
    def __init__(self, dim):
        super().__init__()
        self.mu = nn.Linear(dim, dim)
        self.logvar = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, 1)
        self.noise_mode = "prior_replacement"
        self.balance_mode = 'legacy'
        self.keep_target = .6
        self.balance_weight = 2.
        self.polarization_weight = 1.
        nn.init.eye_(self.mu.weight); nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logvar.weight); nn.init.constant_(self.logvar.bias, -4.)
        nn.init.normal_(self.gate.weight, std=.01); nn.init.zeros_(self.gate.bias)

    def forward(self, x, mask, use_gate=True):
        mu, lv = self.mu(x), self.logvar(x).clamp(-10, 5)
        lv = lv + 2*math.log(getattr(self,"noise_scale",1.))
        gate_input = mu if getattr(self,"gate_on_mu",False) else x
        if getattr(self,"gate_normalize",False):
            gate_input = F.layer_norm(gate_input, (gate_input.shape[-1],))
        logits = self.gate(gate_input)
        if getattr(self, "context_logits", None) is not None:
            logits = logits + self.context_logits
        g = torch.sigmoid(logits) if use_gate else torch.ones_like(x[..., :1])
        if self.training:
            z = mu + torch.exp(.5*lv)*torch.randn_like(mu)
            prior_noise = torch.randn_like(mu)  # Preserve RNG draws across variants.
            out = g*z if self.noise_mode == "gated_posterior" else g*z + (1-g)*prior_noise
        else:
            out = g*mu
        valid = mask.to(x.dtype)
        denom = valid.sum().clamp_min(1)
        kl = (.5*(mu.square()+lv.exp()-1-lv).mean(-1)*valid).sum()/denom
        gate = g.squeeze(-1)
        mean = (gate*valid).sum()/denom
        if getattr(self,'polar_form','quadratic')=='entropy':
            z=logits.squeeze(-1)
            # Normalize maximum entropy to .25, equal to max g(1-g).
            penalty=-(gate*F.logsigmoid(z)+(1-gate)*F.logsigmoid(-z))/(4*math.log(2))
        else:penalty=gate*(1-gate)
        polar = self.polarization_weight * (penalty*valid).sum()/denom
        if self.balance_mode == 'bernoulli' and use_gate:
            # Cross entropy of each sample's mean retention, computed in log
            # space. Unlike squared mean penalties this recovers saturated
            # sigmoid logits even when sigmoid rounds to exactly 0 or 1.
            counts = valid.sum(-1)
            if (counts == 0).any():
                raise ValueError('Budget regularizer requires nonempty samples')
            z = logits.squeeze(-1)
            log_keep = torch.logsumexp(F.logsigmoid(z).masked_fill(~mask.bool(), -torch.inf), -1) - counts.log()
            log_drop = torch.logsumexp(F.logsigmoid(-z).masked_fill(~mask.bool(), -torch.inf), -1) - counts.log()
            budget = -(self.keep_target*log_keep + (1-self.keep_target)*log_drop).mean()
            polar = polar + self.balance_weight * budget
        else:
            polar = polar + (mean-.5).abs().sub(.2).clamp_min(0).square()
        self.last_gate = g.detach()
        # Keep the current gate connected for reliability-aware matching.
        self.current_gate = g
        return out*valid.unsqueeze(-1), kl, polar


class FixedMassTransport(nn.Module):
    """Entropic partial OT via log-domain KL-Dykstra on real edges only.

    row <= uniform a, column <= uniform b, total = s. Finite-iteration
    residuals are reported, not silently treated as exact convergence.
    """
    def __init__(self, eps=.1, n_iters=100, mass=.6):
        super().__init__()
        if eps <= 0 or n_iters < 1 or not .3 < mass < .9:
            raise ValueError('Require eps>0, iterations>=1, .3<mass<.9')
        self.eps, self.n_iters = eps, n_iters
        raw=(mass-.3)/.6
        self.s_logit=nn.Parameter(torch.tensor(math.log(raw/(1-raw))))
        self.warmup=False

    def set_warmup(self, active): self.warmup=bool(active)
    def get_s(self):
        return self.s_logit.new_tensor(1.) if self.warmup else .3+.6*self.s_logit.sigmoid()

    def forward(self, cost, mask_t, mask_v, full=False):
        s=cost.new_tensor(1.) if full else self.get_s().to(cost)
        plans=[]; objectives=[]; residuals=[]
        for c,mt,mv in zip(cost,mask_t.bool(),mask_v.bool()):
            it,iv=mt.nonzero().flatten(),mv.nonzero().flatten()
            if not len(it) or not len(iv): raise ValueError('Empty transport support')
            small=c[it[:,None],iv[None,:]]
            logp=-small/self.eps
            corrections=[torch.zeros_like(logp) for _ in range(3)]
            for _ in range(self.n_iters):
                for axis in range(3):
                    y=logp+corrections[axis]
                    if axis==0:
                        adjustment=(-math.log(len(it))-torch.logsumexp(y,1,keepdim=True)).clamp_max(0)
                    elif axis==1:
                        adjustment=(-math.log(len(iv))-torch.logsumexp(y,0,keepdim=True)).clamp_max(0)
                    else:
                        adjustment=s.log()-torch.logsumexp(y.flatten(),0)
                    logp=y+adjustment
                    corrections[axis]=y-logp
            p=logp.exp()
            # scatter retains autograd while enforcing exact zero padding.
            whole=torch.zeros_like(c).index_put((it[:,None],iv[None,:]),p)
            plans.append(whole)
            objectives.append((p*small).sum()+self.eps*(p*(logp-1)).sum())
            residuals.append(torch.stack([(p.sum(1)-1/len(it)).clamp_min(0).max(),
                (p.sum(0)-1/len(iv)).clamp_min(0).max(),(p.sum()-s).abs()]))
        self.last_residual=torch.stack(residuals).detach()
        return torch.stack(plans),torch.stack(objectives).mean()


class PreviousAligner(nn.Module):
    def __init__(self,text_dim,visual_dim,hidden_dim,eps=.1,n_iters=100,mass=.6,
                 ib='polarized',transport='partial',aggregation='capacity',safe_fusion=False,
                 cost_mode='cosine'):
        super().__init__()
        self.explore = json.loads(os.environ.get('PGMNER_EXPERIMENT', '{}'))
        self.register_buffer('local_fusion_coefficient',torch.tensor(float(self.explore.get('local_coefficient',.1))))
        self.text_proj=nn.Linear(text_dim,hidden_dim)
        self.visual_proj=nn.Linear(visual_dim,hidden_dim)
        # Always initialize identical parameter sets across ablations.
        self.ib_t=LocalPolarizedIB(hidden_dim);self.ib_v=LocalPolarizedIB(hidden_dim)
        self.context_t=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.context_v=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.visual_ln=nn.LayerNorm(hidden_dim)
        self.cross_logit=nn.Parameter(torch.tensor(-1.38629436))
        self.interaction=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.match_t=nn.Linear(hidden_dim,512)
        self.match_v=nn.Linear(visual_dim,512,bias=False)
        if self.explore.get('clip_anchor',False):
            state=torch.load('pretrain/clip-vit-base-patch16/pytorch_model.bin',map_location='cpu',weights_only=True)
            self.match_v.weight.data.copy_(state['visual_projection.weight'])
            del state
            self.match_v.requires_grad_(False)
        self.solver=FixedMassTransport(eps,n_iters,mass)
        self.ib,self.transport,self.aggregation=ib,transport,aggregation
        self.safe_fusion = safe_fusion
        if cost_mode not in ('cosine', 'gate_reliability'): raise ValueError(cost_mode)
        self.cost_mode = cost_mode
        if safe_fusion:
            self.fusion_logit = nn.Parameter(torch.tensor(math.log(.2/.8)))
        if ib not in ('off','vib','polarized'):raise ValueError(ib)
        if transport not in ('partial','full','uniform','uniform_partial','none'):raise ValueError(transport)
        if aggregation not in ('capacity','raw'):raise ValueError(aggregation)

    def set_partial_ot_warmup(self,active):self.solver.set_warmup(active)

    def forward(self,Z_t,Z_v,mask_t=None,mask_v=None,**kwargs):
        self.ib_t.keep_target=float(self.explore.get('text_keep',.85))
        self.ib_v.keep_target=float(self.explore.get('image_keep',.6))
        self.ib_t.noise_scale=self.ib_v.noise_scale=float(self.explore.get('noise_scale',.25))
        t,v=self.text_proj(Z_t),self.visual_proj(Z_v)
        mt=torch.ones_like(t[...,0]) if mask_t is None else mask_t.to(t)
        mv=torch.ones_like(v[...,0]) if mask_v is None else mask_v.to(v)
        text_base=t*mt.unsqueeze(-1)
        image_valid=Z_v.abs().sum((1,2)).gt(0)
        v=v*image_valid[:,None,None]
        pooled_t=(t*mt.unsqueeze(-1)).sum(1)/mt.sum(1,keepdim=True).clamp_min(1)
        pooled_v=(v*mv.unsqueeze(-1)).sum(1)/mv.sum(1,keepdim=True).clamp_min(1)
        # Cross-modal conditioning is bounded and padding-aware.
        self.ib_t.context_logits=2*(F.normalize(t,dim=-1)*F.normalize(self.context_t(pooled_v),dim=-1)[:,None,:]).sum(-1,keepdim=True)
        self.ib_v.context_logits=2*(F.normalize(v,dim=-1)*F.normalize(self.context_v(pooled_t),dim=-1)[:,None,:]).sum(-1,keepdim=True)
        if not self.explore.get('context', True):
            self.ib_t.context_logits=None; self.ib_v.context_logits=None
        aux=t.new_zeros(())
        valid_ids=image_valid.nonzero().flatten()
        if len(valid_ids)>1:
            qt=F.normalize(self.match_t(pooled_t[valid_ids]),dim=-1)
            qv=F.normalize(self.match_v(Z_v[valid_ids,0,:]),dim=-1)
            scores=qt@qv.T/.1
            targets=torch.arange(len(valid_ids),device=t.device)
            aux=(F.cross_entropy(scores,targets)+F.cross_entropy(scores.T,targets))/2
        aux=aux*float(self.explore.get('contrastive', .03))
        kl=polar=t.new_zeros(())
        if self.ib!='off':
            t,kt,pt=self.ib_t(t,mt,self.ib=='polarized')
            v,kv,pv=self.ib_v(v,mv,self.ib=='polarized')
            side=self.explore.get('ib_side','both')
            if side=='text':
                v=self.visual_proj(Z_v)*image_valid[:,None,None];kv=kv*0;pv=pv*0
                self.ib_v.current_gate=torch.ones_like(v[...,:1]);self.ib_v.last_gate=self.ib_v.current_gate.detach()
            if side=='visual':
                t=text_base;kt=kt*0;pt=pt*0
                self.ib_t.current_gate=torch.ones_like(t[...,:1]);self.ib_t.last_gate=self.ib_t.current_gate.detach()
            kl=(kt+kv)/2
            if self.ib=='polarized':polar=pt+pv
        else:
            t=t*mt.unsqueeze(-1);v=v*mv.unsqueeze(-1)
        if self.explore.get('clip_anchor',False):
            # Contrastive training and transport share this text projection.
            cost=1-torch.bmm(F.normalize(self.match_t(text_base),dim=-1),F.normalize(self.match_v(Z_v),dim=-1).transpose(1,2))
        else:
            cost=1-torch.bmm(F.normalize(t,dim=-1),F.normalize(v,dim=-1).transpose(1,2))
        if self.cost_mode == 'gate_reliability' and self.ib == 'polarized':
            reliability = self.ib_t.current_gate * self.ib_v.current_gate.transpose(1,2)
            # Expected cost: reliable pair uses cosine distance; unreliable
            # pair uses its maximum cost (2). Scalar gates survive normalization.
            cost = cost + float(self.explore.get("gate_cost",.15))*(1-reliability)
        # One detached linearization of squared intra-modal distance distortion.
        # This is an approximate FGW-inspired refinement, not an exact FGW solver.
        structure=cost.new_zeros(cost.shape)
        graph_weight=float(self.explore.get('structure', .3))
        if graph_weight and self.transport in ('partial','full'):
            with torch.no_grad():
                nt,nv=F.normalize(t,dim=-1),F.normalize(v,dim=-1)
                dt=1-torch.bmm(nt,nt.transpose(1,2))
                dv=1-torch.bmm(nv,nv.transpose(1,2))
                dt=dt*mt[:,:,None]*mt[:,None,:]
                dv=dv*mv[:,:,None]*mv[:,None,:]
                anchor=torch.softmax((-cost/.1).masked_fill(~mv.bool()[:,None,:],-1e4),-1)
                anchor=anchor*mt[:,:,None]/mt.sum(1)[:,None,None].clamp_min(1)
                rows,cols=anchor.sum(2),anchor.sum(1)
                structure=(dt.square()@rows.unsqueeze(-1))+(dv.square()@cols.unsqueeze(-1)).transpose(1,2)-2*torch.bmm(torch.bmm(dt,anchor),dv.transpose(1,2))
                structure=structure.clamp_min(0)
            cost=cost+graph_weight*structure
        a=mt/mt.sum(1,keepdim=True).clamp_min(1)
        b=mv/mv.sum(1,keepdim=True).clamp_min(1)
        if self.transport in ('partial','full'):
            plan,loss=self.solver(cost,mt,mv,full=self.transport=='full')
        elif self.transport=='uniform_partial':
            plan=self.solver.get_s()*a.unsqueeze(-1)*b.unsqueeze(-2)
            loss=cost.new_zeros(())
        elif self.transport=='uniform':
            plan=a.unsqueeze(-1)*b.unsqueeze(-2);loss=cost.new_zeros(())
        else:
            plan=torch.zeros_like(cost);loss=cost.new_zeros(())
        # Divide by capacity a_i, NOT matched row mass: preserve rejection amplitude.
        at=torch.bmm(plan,v);av=torch.bmm(plan.transpose(1,2),t)
        if self.aggregation=='capacity':
            at=at/a.clamp_min(1e-12).unsqueeze(-1)
            av=av/b.clamp_min(1e-12).unsqueeze(-1)
        if self.safe_fusion:
            mix=float(self.explore.get("text_mix",.2))
            t = (1-mix)*text_base + mix*t
            if self.explore.get('normalize_fusion', True):
                occupancy=(plan.sum(-1)/a.clamp_min(1e-12)).clamp(0,1)
                at=self.visual_ln(at)*occupancy.unsqueeze(-1)*self.cross_logit.sigmoid()
            else:
                at=at*(.5*self.fusion_logit.sigmoid())
        if self.explore.get('interaction',False):
            # Content-dependent residual, with zero contribution when transport is disabled.
            occupancy=(plan.sum(-1)/a.clamp_min(1e-12)).clamp(0,1)
            local=torch.bmm(plan,v)/plan.sum(-1).clamp_min(1e-8).unsqueeze(-1)
            content=.75*local+.25*pooled_v[:,None,:]
            pair=F.layer_norm(text_base,(text_base.shape[-1],))*F.layer_norm(content,(content.shape[-1],))
            at=at+.05*self.interaction(pair)*occupancy.unsqueeze(-1)
        at=at*image_valid[:,None,None]
        comp={'L_sem':(plan*cost).sum((1,2)).mean(), 'L_str':(plan*structure).sum((1,2)).mean(),
              'L_reg':(plan*(plan.clamp_min(1e-30).log()-1)).sum((1,2)).mean(),
              'repair_kl':kl,'repair_polar':polar,'explore_aux':aux}
        with torch.no_grad():
            self.last_diagnostics = {
                's': float(self.solver.get_s()), 'real_mass': float(plan.sum((1,2)).mean()),
                'cost_mean': float(cost.mean()), 'contrastive_loss':float(aux), 'cross_coefficient':float(self.cross_logit.sigmoid()), 'valid_image_fraction':float(image_valid.float().mean()), 'cost_std': float(cost.std()),
                'text_norm': float(t.norm(dim=-1)[mt.bool()].mean()),
                'aligned_visual_norm': float(at.norm(dim=-1)[mt.bool()].mean()),
                'visual_norm': float(v.norm(dim=-1)[mv.bool()].mean()),
            }
            if self.safe_fusion:
                self.last_diagnostics['fusion_coefficient'] = float(.5*self.fusion_logit.sigmoid())
            if self.transport in ('partial','full'):
                self.last_diagnostics['max_capacity_mass_residual'] = float(self.solver.last_residual.max())
            if self.ib != 'off':
                self.last_diagnostics['text_gate_mean'] = float(self.ib_t.last_gate.squeeze(-1)[mt.bool()].mean())
                self.last_diagnostics['image_gate_mean'] = float(self.ib_v.last_gate.squeeze(-1)[mv.bool()].mean())
                for name, ib, mask in [('text', self.ib_t, mt), ('image', self.ib_v, mv)]:
                    gates = ib.last_gate.squeeze(-1)[mask.bool()]
                    self.last_diagnostics[name+'_gate_low_fraction'] = float((gates < .1).float().mean())
                    self.last_diagnostics[name+'_gate_high_fraction'] = float((gates > .9).float().mean())
                    self.last_diagnostics[name+'_gate_std'] = float(gates.std(unbiased=False))
        self.last_gamma=plan.detach().cpu()
        return t,v,at,av,plan,loss,comp


class RepairedAligner(PreviousAligner):
    """Shared common path; gate and transport only the centered local residual."""
    def set_local_epoch(self, epoch):
        if self.explore.get('local_schedule') == 'ramp':
            values=(.1,.1,.2,.3,.4,.5,.5,.5)
            self.local_fusion_coefficient.fill_(values[min(7,max(0,int(epoch)-1))])

    def forward(self,Z_t,Z_v,mask_t=None,mask_v=None,**kwargs):
        t=self.text_proj(Z_t);v=self.visual_proj(Z_v)
        mt=torch.ones_like(t[...,0]) if mask_t is None else mask_t.to(t)
        mv=torch.ones_like(v[...,0]) if mask_v is None else mask_v.to(v)
        valid=Z_v.abs().sum((1,2)).gt(0)
        # CLS is excluded from local OT and patch statistics.
        if v.shape[1]>1:mv=mv.clone();mv[:,0]=0
        denom=mv.sum(1,keepdim=True).clamp_min(1)
        center=(v*mv[...,None]).sum(1,keepdim=True)/denom[...,None]
        centered=(v-center)*mv[...,None]
        self.ib_v.gate_on_mu=bool(self.explore.get('fix_mu_gate',False))
        self.ib_v.polar_form=self.explore.get('polar_form','quadratic')
        self.ib_v.keep_target=float(self.explore.get('image_keep',.6))
        self.ib_v.noise_scale=float(self.explore.get('noise_scale',.25))
        self.ib_v.gate_normalize=bool(self.explore.get('gate_normalize',False))
        self.ib_v.context_logits=None
        if self.explore.get('gate_context',False):
            pooled_context=(t*mt[...,None]).sum(1)/mt.sum(1,keepdim=True).clamp_min(1)
            context=F.normalize(self.context_t(pooled_context),dim=-1)
            patches=F.normalize(self.ib_v.mu(centered),dim=-1)
            self.ib_v.context_logits=2.*(patches*context[:,None,:]).sum(-1,keepdim=True)
        local,kl,polar=self.ib_v(centered,mv,self.ib=='polarized')
        if self.ib!='polarized':polar=polar*0
        g=self.ib_v.current_gate
        # Remove affine bias from the residual; retain it in the common path.
        bias=self.ib_v.mu.bias[None,None,:]
        local=local-g*bias*mv[...,None]
        if self.ib=='off':
            local=(self.ib_v.mu(centered)-bias)*mv[...,None];kl=kl*0;polar=polar*0
        self.ib_t.current_gate=torch.ones_like(t[...,:1]);self.ib_t.last_gate=self.ib_t.current_gate.detach()
        common=self.ib_v.mu(center)
        gain=float(self.explore.get('residual_gain',1.))
        local=local*min(max(gain,0.),4.)
        cost_t,cost_v=t,local
        if self.explore.get('fix_centered_cost',False):
            cost_t=(t-(t*mt[...,None]).sum(1,keepdim=True)/mt.sum(1,keepdim=True).clamp_min(1)[...,None])*mt[...,None]
            cost_v=(local-(local*mv[...,None]).sum(1,keepdim=True)/denom[...,None])*mv[...,None]
        cost=1-torch.bmm(F.normalize(cost_t,dim=-1),F.normalize(cost_v,dim=-1).transpose(1,2))
        if self.ib=='polarized':cost=cost+.15*(1-g.transpose(1,2))
        a=mt/mt.sum(1,keepdim=True).clamp_min(1);b=mv/denom
        if self.transport in ('partial','full'):plan,loss=self.solver(cost,mt,mv,full=self.transport=='full')
        elif self.transport=='uniform_partial':plan=self.solver.get_s()*a[:,:,None]*b[:,None,:];loss=t.new_zeros(())
        else:plan=cost*0;loss=t.new_zeros(())
        aligned=torch.bmm(plan,local)/a.clamp_min(1e-12)[...,None]
        self.local_aux_features=aligned
        self.local_aux_text=t
        self.local_aux_valid=valid
        coefficient=self.local_fusion_coefficient
        if self.explore.get('selected_common',False):
            # Selection also controls the image summary; normalize retained mass
            # so this tests region composition rather than gate amplitude.
            weights=g*mv[...,None]
            residual_mean=self.ib_v.mu(centered)-bias
            common=common+(weights*residual_mean).sum(1,keepdim=True)/weights.sum(1,keepdim=True).clamp_min(1e-6)
        self.common_features=common
        common_coefficient=float(self.explore.get('common_coefficient',.1))
        at=(common_coefficient*common+coefficient*aligned)*mt[...,None]*valid[:,None,None]
        pooled=(t*mt[...,None]).sum(1)/mt.sum(1,keepdim=True).clamp_min(1)
        aux=t.new_zeros(());ix=valid.nonzero().flatten()
        if len(ix)>1:
            scores=F.normalize(self.match_t(pooled[ix]),dim=-1)@F.normalize(self.match_v(Z_v[ix,0]),dim=-1).T/.1
            target=torch.arange(len(ix),device=t.device)
            aux=.5*(F.cross_entropy(scores,target)+F.cross_entropy(scores.T,target))*float(self.explore.get('contrastive',.03))
        comp={'L_sem':(plan*cost).sum((1,2)).mean(),'L_str':t.new_zeros(()),'L_reg':t.new_zeros(()),'repair_kl':kl,'repair_polar':polar,'explore_aux':aux}
        with torch.no_grad():
            gates=g.squeeze(-1)[mv.bool()]
            self.last_diagnostics={'image_gate_mean':float(gates.mean()),'image_gate_std':float(gates.std(unbiased=False)),'image_gate_high_fraction':float((gates>.99).float().mean()),'common_norm':float(common.norm(dim=-1).mean()),'local_norm':float(aligned.norm(dim=-1)[mt.bool()].mean()),'s':float(self.solver.get_s()),'real_mass':float(plan.sum((1,2)).mean())}
        self.last_gamma=plan.detach().cpu()
        return t*mt[...,None],local,at,torch.bmm(plan.transpose(1,2),t),plan,loss,comp
