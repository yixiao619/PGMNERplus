"""Extended controlled mechanism analyses for the journal manuscript."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from models.repaired_alignment import FixedMassTransport
torch.set_num_threads(1)
torch.manual_seed(42)
PROTOCOL=json.loads((HERE/'protocol.json').read_text())

def solve(cost,eps=.1,mass=.6,mt=None,mv=None):
    mt=torch.ones(1,cost.shape[0]) if mt is None else mt
    mv=torch.ones(1,cost.shape[1]) if mv is None else mv
    for iterations in PROTOCOL['solver']['iteration_attempts']:
        solver=FixedMassTransport(eps=eps,n_iters=iterations,mass=mass).double()
        with torch.no_grad(): p,_=solver(cost[None],mt,mv)
        residual=float(solver.last_residual.max())
        if residual<PROTOCOL['solver']['tolerance']:break
    return p[0],dict(iterations=iterations,residual=residual,converged=residual<1e-7)

def saturation():
    settings=PROTOCOL['expanded_A'];initial=settings['initial_logits']
    configurations=[('quadratic_band',.6,z) for z in initial]
    configurations += [('retention_budget',target,z) for target in settings['targets'] for z in initial]
    z=torch.nn.Parameter(torch.tensor([c[2] for c in configurations],dtype=torch.float64))
    target=torch.tensor([c[1] for c in configurations],dtype=torch.float64)
    budget=torch.tensor([c[0]=='retention_budget' for c in configurations])
    opt=torch.optim.SGD([z],lr=settings['lr'])
    trace=[]
    for step in range(settings['steps']+1):
        g=z.sigmoid()
        penalty=torch.where(budget,2*(-target*F.logsigmoid(z)-(1-target)*F.logsigmoid(-z)),
                            (g-.5).abs().sub(.2).clamp_min(0).square())
        loss=g*(1-g)+penalty
        opt.zero_grad();loss.sum().backward()
        trace.append({'step':step,'gate':g.detach().tolist(),'gradient':z.grad.tolist()})
        if step<settings['steps']:opt.step()
    result=[]
    for i,(mode,pi,init) in enumerate(configurations):
        crossing=next((t['step'] for t in trace if .1<=t['gate'][i]<=.9),None)
        result.append(dict(mode=mode,target=pi,initial_logit=init,recovery_step=crossing,
                           initial_gradient=trace[0]['gradient'][i],final_gate=trace[-1]['gate'][i]))
    return dict(configurations=result,trace=trace)

def cost_matrix(bad,gap=1.):
    # Each distractor copies a particular prototype direction with cosine 1-gap
    # and has its own orthogonal component. At gap=0 it is unidentifiable.
    good=6;basis=torch.eye(good+bad,dtype=torch.float64)
    patches=[basis[i] for i in range(good)]
    for j in range(bad):
        patches.append((1-gap)*basis[j%good]+np.sqrt(1-(1-gap)**2)*basis[good+j])
    return 1-basis[:good]@torch.stack(patches).T

def gap_temperature():
    settings=PROTOCOL['E_gap_temperature'];records=[]
    for eps in settings['eps']:
        for gap in settings['gaps']:
            p,info=solve(cost_matrix(settings['distractors'],gap),eps,settings['mass'])
            mass=float(p.sum())
            records.append(dict(eps=eps,gap=gap,gap_over_eps=gap/eps,
                distractor_fraction=float(p[:,6:].sum())/mass,
                correct_fraction=float(p[:,:6].diag().sum())/mass,**info))
    return records

def budgets():
    settings=PROTOCOL['F_budget_capacity'];records=[]
    for bad in settings['distractors']:
        for mass in settings['mass']:
            p,info=solve(cost_matrix(bad),settings['eps'],mass)
            actual=float(p.sum());lower=max(0.,actual-6/(6+bad))
            noise=float(p[:,6:].sum())
            records.append(dict(distractors=bad,mass=mass,actual_mass=actual,
                distractor_mass=noise,lower_bound=lower,excess_over_bound=noise-lower,
                distractor_fraction=noise/actual,**info))
    return records

def stability():
    settings=PROTOCOL['G_stability'];records=[]
    base=.3+.4*torch.rand(*settings['matrix'],dtype=torch.float64)
    directions=[]
    for i in range(settings['directions']):
        d=2*torch.rand_like(base)-1;d=d/d.abs().max();directions.append(d)
    for eps in settings['eps']:
        reference,reference_info=solve(base,eps,settings['mass'])
        for delta in settings['delta']:
            for index,d in enumerate(directions):
                p,info=solve(base+delta*d,eps,settings['mass'])
                norm=float((p-reference).abs().sum())
                actual=float(reference.sum())
                bound=actual*min(2.,delta/eps)
                records.append(dict(eps=eps,delta=delta,direction=index,
                    l1_difference=norm,normalized_l1=norm/actual,bound=bound,
                    bound_ratio=norm/bound,reference_residual=reference_info['residual'],**info))
    return records

def invariance():
    settings=PROTOCOL['H_invariance'];records=[]
    for trial in range(settings['trials']):
        n,m=settings['matrix'];cost=torch.rand(n,m,dtype=torch.float64)
        p,info=solve(cost,settings['eps'],settings['mass'])
        rows=torch.randperm(n);cols=torch.randperm(m)
        perm,pi=solve(cost[rows][:,cols],settings['eps'],settings['mass'])
        nr,nc=settings['append_padding']
        # Arbitrary invalid costs include very attractive negative values.
        padded=-100*torch.rand(n+nr,m+nc,dtype=torch.float64)
        padded[:n,:m]=cost
        mt=torch.cat([torch.ones(n),torch.zeros(nr)])[None]
        mv=torch.cat([torch.ones(m),torch.zeros(nc)])[None]
        pp,pa=solve(padded,settings['eps'],settings['mass'],mt,mv)
        records.append(dict(trial=trial,permutation_max_error=float((perm-p[rows][:,cols]).abs().max()),
            padding_valid_max_error=float((pp[:n,:m]-p).abs().max()),
            padded_mass=float(pp[n:,:].sum()+pp[:n,m:].sum()),
            max_residual=max(info['residual'],pi['residual'],pa['residual']),
            converged=info['converged'] and pi['converged'] and pa['converged']))
    return records

if __name__=='__main__':
    output=HERE/'results.json'
    if output.exists():raise SystemExit('Refuse to overwrite recorded results')
    result={}
    for key,fn in [('A_expanded',saturation),('E',gap_temperature),('F',budgets),('G',stability),('H',invariance)]:
        result[key]=fn();(HERE/f'{key}.json').write_text(json.dumps(result[key],indent=2))
        print(key,'complete',flush=True)
    output.write_text(json.dumps(result,indent=2))
    (HERE/'source_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve(),ROOT/'models/repaired_alignment.py']},indent=2))
