import sys,os,json,importlib.util,inspect,torch
from models.repaired_alignment import LocalPolarizedIB,RepairedAligner,FixedMassTransport
spec=importlib.util.spec_from_file_location('prior','/home/jyx/PGMNER_ppib_ramp/models/repaired_alignment.py');prior=importlib.util.module_from_spec(spec);sys.modules['prior']=prior;spec.loader.exec_module(prior)
assert inspect.getsource(FixedMassTransport)==inspect.getsource(prior.FixedMassTransport)
os.environ['PGMNER_EXPERIMENT']=json.dumps({'fix_mu_gate':True,'local_coefficient':.3})
torch.manual_seed(42)
a=prior.RepairedAligner(12,8,16,n_iters=25).eval();b=RepairedAligner(12,8,16,n_iters=25).eval();b.load_state_dict(a.state_dict())
t=torch.randn(2,5,12);v=torch.randn(2,7,8)
x=a(t,v);y=b(t,v);assert torch.equal(x[2],y[2]) and torch.equal(x[4],y[4])
ib=LocalPolarizedIB(16).eval();ib.gate_on_mu=True;ib.gate_normalize=True
z=torch.randn(2,7,16);mask=torch.ones(2,7)
ib(z,mask);g=ib.current_gate.clone();ib(z*5,mask);assert torch.allclose(g,ib.current_gate,atol=1e-5)
for norm,context in [(True,False),(False,True),(True,True)]:
 b.explore.update(gate_normalize=norm,gate_context=context)
 b.zero_grad();out=b(t,v);loss=out[2].square().mean()+.001*out[6]['repair_polar'];loss.backward()
 assert b.ib_v.gate.weight.grad.abs().sum()>0
 assert all(torch.isfinite(p.grad).all() for p in b.parameters() if p.grad is not None)
 if context:
  assert b.context_t.weight.grad.abs().sum()>0
  gates=b.ib_v.current_gate.detach().clone();b(t.flip(0),v);assert not torch.allclose(gates,b.ib_v.current_gate)
print('PASS default equivalence, unchanged GOTA solver, scale robustness, text-conditioned gates, finite gradients')
