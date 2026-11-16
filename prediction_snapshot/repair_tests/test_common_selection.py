import sys,os,json,torch,importlib.util
from models.repaired_alignment import RepairedAligner
spec=importlib.util.spec_from_file_location('prior','/home/jyx/PGMNER_ppib_select/models/repaired_alignment.py');old=importlib.util.module_from_spec(spec);sys.modules['prior']=old;spec.loader.exec_module(old)
os.environ['PGMNER_EXPERIMENT']=json.dumps({'local_coefficient':.3,'gate_context':True,'fix_mu_gate':True})
torch.manual_seed(42);a=old.RepairedAligner(12,8,16,n_iters=25).eval();b=RepairedAligner(12,8,16,n_iters=25).eval();b.load_state_dict(a.state_dict());t=torch.randn(2,5,12);v=torch.randn(2,7,8)
x=a(t,v);y=b(t,v);assert torch.equal(x[2],y[2]);assert torch.equal(x[4],y[4])
for selected,coef in [(False,.03),(True,.1),(True,.03)]:
 b.explore.update(selected_common=selected,common_coefficient=coef);b.zero_grad();z=b(t,v)
 assert torch.equal(x[4],z[4]) and torch.equal(x[1],z[1]),'GOTA/local path changed'
 z[2].square().mean().backward(retain_graph=True);assert b.ib_v.gate.weight.grad.abs().sum()>0
 if selected:
  gg=torch.autograd.grad(b.common_features.square().mean(),b.ib_v.gate.weight)[0];assert torch.isfinite(gg).all() and gg.abs().sum()>0
 assert all(torch.isfinite(p.grad).all() for p in b.parameters() if p.grad is not None)
 invalid=b(t,torch.zeros_like(v));assert torch.count_nonzero(invalid[2])==0
b.explore.update(selected_common=False,common_coefficient=.1,gate_context=False)
with torch.no_grad():b.ib_v.gate.weight.zero_();b.ib_v.gate.bias.zero_()
x=b(t,v);b.explore['selected_common']=True;y=b(t,v);assert torch.allclose(x[2],y[2],atol=1e-6),'Uniform gate must recover original mean'
print('PASS baseline equivalence, identical GOTA/local values, finite gate gradients, missing image mask, uniform gate equivalence')
