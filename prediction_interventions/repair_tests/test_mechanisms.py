import ast,copy,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.repaired_alignment import RepairedAligner,FixedMassTransport

torch.set_num_threads(1);torch.manual_seed(42)
report={}
# Execute the real optimizer function without importing unrelated training dependencies.
root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/'engine_utils.py').read_text())
func=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='prepare_optimizer_scheduler')
ns={'AdamW':torch.optim.AdamW,'get_linear_schedule_with_warmup':lambda *a,**k:None}
exec(compile(ast.Module(body=[func],type_ignores=[]),'engine_utils.py','exec'),ns)
model=nn.Sequential(nn.Linear(4,4),nn.LayerNorm(4))
opt,_=ns[func.name](SimpleNamespace(weight_decay=.01,lr=.01,adam_epsilon=1e-8,warmup_ratio=.1),model,10)
ids=[id(p) for g in opt.param_groups for p in g['params']]
assert len(ids)==len(set(ids))==len(list(model.parameters()))
report['optimizer_parameters_covered']=len(ids)
# Historical function demonstrates the actual omitted parameters.
oldtree=ast.parse(Path('/data/jyx/yixiao/MINER-My/engine_utils.py').read_text())
oldfun=next(x for x in oldtree.body if isinstance(x,ast.FunctionDef) and x.name==func.name)
exec(compile(ast.Module(body=[oldfun],type_ignores=[]),'original','exec'),ns)
oldopt,_=ns[func.name](SimpleNamespace(weight_decay=.01,lr=.01,adam_epsilon=1e-8,warmup_ratio=.1),model,10)
report['historical_optimizer_parameters_covered']=sum(len(g['params']) for g in oldopt.param_groups)
# Task loss alone must train local gates, posterior and transport cost projections.
a=RepairedAligner(8,6,8,n_iters=100)
x=torch.randn(2,4,8);v=torch.randn(2,5,6);mt=torch.tensor([[1,1,1,0],[1,1,1,1.]])
mv=torch.tensor([[1,1,1,1,0],[1,1,1,1,1.]])
fusion=nn.Linear(16,8);norm=nn.LayerNorm(8);head=nn.Linear(8,3)
def logits(align,x=x,v=v,mask=mt):
 t,_,at,_,p,loss,comp=align(x,v,mask,mv)
 return head(norm(fusion(torch.cat([t,at],-1))+at)),p,loss,comp
z,p,loss,comp=logits(a)
labels=torch.tensor([[0,1,2,0],[2,1,0,1]])
task=nn.functional.cross_entropy(z[mt.bool()],labels[mt.bool()]);task.backward()
for name in ['ib_t.gate.weight','ib_v.gate.weight','ib_t.mu.weight','ib_v.mu.weight','ib_t.logvar.weight','ib_v.logvar.weight','text_proj.weight','visual_proj.weight','solver.s_logit']:
 g=dict(a.named_parameters())[name].grad
 assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0,name
 report['task_gradient_'+name]=float(g.norm())
# Exact zero padding, budget, capacity, non-default s initialization.
assert p[0,3].abs().max()==0 and p[0,:,4].abs().max()==0
assert torch.allclose(p.sum((1,2)),a.solver.get_s().expand(2),atol=1e-5)
assert a.solver.last_residual.max()<1e-4
assert abs(float(FixedMassTransport(mass=.5).get_s())-.5)<1e-6
report['max_solver_residual']=float(a.solver.last_residual.max())
# Comparison with independent constrained optimizer on a small convex problem.
import numpy as np
from scipy.optimize import minimize
c=np.array([[.1,.8,.7],[.6,.2,.9]])
solver=FixedMassTransport(eps=.2,n_iters=300).double()
cp=torch.tensor(c).unsqueeze(0)
pred,_=solver(cp,torch.ones(1,2),torch.ones(1,3))
def objective(y):return float((y*c.flatten()).sum()+.2*(y*(np.log(y)-1)).sum())
cons=[{'type':'eq','fun':lambda y:y.sum()-.6},
 {'type':'ineq','fun':lambda y:.5-y.reshape(2,3).sum(1)},
 {'type':'ineq','fun':lambda y:1/3-y.reshape(2,3).sum(0)}]
r=minimize(objective,np.full(6,.1),method='SLSQP',constraints=cons,bounds=[(1e-12,None)]*6,options={'ftol':1e-12,'maxiter':1000})
assert r.success
err=np.max(np.abs(pred.detach().numpy().flatten()-r.x));assert err<2e-5
report['independent_solver_max_plan_error']=float(err)
# Deterministic inference and controlled interventions.
a.eval();base=logits(a)[0];assert torch.equal(base,logits(a)[0])
with torch.no_grad():
 a.ib_t.gate.bias.add_(3)
drift=(base-logits(a)[0]).abs().max();assert drift>1e-4
report['gate_intervention_max_logit_change']=float(drift)
a.transport='none';z0=logits(a)[0];z1=logits(a,v=v.flip(0))[0]
assert torch.equal(z0,z1),'no-OT must remove image dependence of logits'
a.transport='partial';z0=logits(a)[0];z1=logits(a,v=v.flip(0))[0]
assert (z0-z1).abs().max()>1e-5
report['image_intervention_max_logit_change']=float((z0-z1).abs().max())
# Appending padded spans must not dilute visual aggregation or change valid logits.
a.ib='off';base=logits(a)[0]
xpad=torch.cat([x,torch.randn(2,3,8)],1);mpad=torch.cat([mt,torch.zeros(2,3)],1)
padded=logits(a,x=xpad,mask=mpad)[0][:,:4]
assert torch.allclose(base[mt.bool()],padded[mt.bool()],atol=2e-6)
for mode in ['full','uniform','none']:
 a.transport=mode
 _,plan,_,_=logits(a)
 target=0 if mode=='none' else 1
 assert torch.allclose(plan.sum((1,2)),torch.full((2,),float(target)),atol=1e-5)
report['tests']='PASS: optimizer coverage, task gradients, constraints, independent optimizer, deterministic eval, gate/image intervention, padding, ablation modes'
(root/'repair_results/mechanism_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
