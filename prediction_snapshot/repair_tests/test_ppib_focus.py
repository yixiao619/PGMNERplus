import os,json,torch,importlib.util
from models.repaired_alignment import LocalPolarizedIB,RepairedAligner
ib=LocalPolarizedIB(8);ib.eval();x=torch.randn(2,5,8);mask=torch.ones(2,5)
with torch.no_grad():ib.gate.weight.zero_();ib.gate.bias.zero_()
ib.polar_form='entropy';assert torch.allclose(ib(x,mask)[2],torch.tensor(.25))
ib.balance_mode='bernoulli'
for bias in [-100.,-20.,0.,20.,100.]:
 ib.zero_grad()
 with torch.no_grad():ib.gate.bias.fill_(bias)
 out=ib(x,mask);out[2].backward();assert torch.isfinite(out[2]) and torch.isfinite(ib.gate.bias.grad).all()
for target,form in [(.8,'quadratic'),(.6,'entropy'),(.8,'entropy')]:
 os.environ['PGMNER_EXPERIMENT']=json.dumps({'image_keep':target,'polar_form':form,'fix_mu_gate':True})
 m=RepairedAligner(12,8,16,n_iters=25,safe_fusion=True);m.ib_v.balance_mode='bernoulli';m.ib_v.noise_mode='gated_posterior'
 out=m(torch.randn(2,5,12),torch.randn(2,7,8));loss=out[2].square().mean()+.001*out[6]['repair_polar'];loss.backward()
 assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 assert m.ib_v.gate.weight.grad.abs().sum()>0
 print('PASS',target,form)
spec=importlib.util.spec_from_file_location('prior','/home/jyx/PGMNER_functional_2x2/models/repaired_alignment.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
os.environ['PGMNER_EXPERIMENT']='{}';a=prior.RepairedAligner(12,8,16,n_iters=25,safe_fusion=True).eval();b=RepairedAligner(12,8,16,n_iters=25,safe_fusion=True).eval();b.load_state_dict(a.state_dict());t=torch.randn(2,5,12);v=torch.randn(2,7,8)
with torch.no_grad():
 aa=a(t,v);bb=b(t,v);assert torch.equal(aa[2],bb[2]);assert torch.equal(aa[6]['repair_polar'],bb[6]['repair_polar'])
print('PASS baseline equivalence, entropy scale and saturated-logit stability')
