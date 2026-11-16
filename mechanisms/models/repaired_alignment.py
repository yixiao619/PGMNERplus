"""Opt-in experimental repair. No historical checkpoint compatibility claim."""
import math
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
        logits = self.gate(x)
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
        polar = self.polarization_weight * ((gate*(1-gate))*valid).sum()/denom
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


class RepairedAligner(nn.Module):
    def __init__(self,text_dim,visual_dim,hidden_dim,eps=.1,n_iters=100,mass=.6,
                 ib='polarized',transport='partial',aggregation='capacity',safe_fusion=False,
                 cost_mode='cosine'):
        super().__init__()
        self.text_proj=nn.Linear(text_dim,hidden_dim)
        self.visual_proj=nn.Linear(visual_dim,hidden_dim)
        # Always initialize identical parameter sets across ablations.
        self.ib_t=LocalPolarizedIB(hidden_dim);self.ib_v=LocalPolarizedIB(hidden_dim)
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
        t,v=self.text_proj(Z_t),self.visual_proj(Z_v)
        mt=torch.ones_like(t[...,0]) if mask_t is None else mask_t.to(t)
        mv=torch.ones_like(v[...,0]) if mask_v is None else mask_v.to(v)
        text_base=t*mt.unsqueeze(-1)
        kl=polar=t.new_zeros(())
        if self.ib!='off':
            t,kt,pt=self.ib_t(t,mt,self.ib=='polarized')
            v,kv,pv=self.ib_v(v,mv,self.ib=='polarized')
            kl=(kt+kv)/2
            if self.ib=='polarized':polar=pt+pv
        else:
            t=t*mt.unsqueeze(-1);v=v*mv.unsqueeze(-1)
        cost=1-torch.bmm(F.normalize(t,dim=-1),F.normalize(v,dim=-1).transpose(1,2))
        if self.cost_mode == 'gate_reliability' and self.ib == 'polarized':
            reliability = self.ib_t.current_gate * self.ib_v.current_gate.transpose(1,2)
            # Expected cost: reliable pair uses cosine distance; unreliable
            # pair uses its maximum cost (2). Scalar gates survive normalization.
            cost = reliability * cost + (1-reliability) * 2
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
            t = .9*text_base + .1*t
            at = at*(.5*self.fusion_logit.sigmoid())
        comp={'L_sem':(plan*cost).sum((1,2)).mean(), 'L_str':cost.new_zeros(()),
              'L_reg':(plan*(plan.clamp_min(1e-30).log()-1)).sum((1,2)).mean(),
              'repair_kl':kl,'repair_polar':polar}
        with torch.no_grad():
            self.last_diagnostics = {
                's': float(self.solver.get_s()), 'real_mass': float(plan.sum((1,2)).mean()),
                'cost_mean': float(cost.mean()), 'cost_std': float(cost.std()),
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
