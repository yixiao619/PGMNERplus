import os,json,torch
from models.repaired_alignment import RepairedAligner
os.environ['PGMNER_EXPERIMENT']=json.dumps({'clip_anchor':False})
torch.manual_seed(42)
m=RepairedAligner(12,8,16,n_iters=30,safe_fusion=True);m.ib_v.balance_mode='bernoulli';m.ib_v.noise_mode='gated_posterior'
t=torch.randn(2,5,12);v=torch.randn(2,7,8);mask=torch.tensor([[1,1,1,0,0],[1,1,1,1,1.]])
out=m(t,v,mask_t=mask);loss=out[2].square().mean()+out[6]['explore_aux']+.001*out[6]['repair_polar'];loss.backward()
for key in ['ib_v.mu.weight','ib_v.gate.weight','visual_proj.weight','solver.s_logit']:
 g=dict(m.named_parameters())[key].grad;assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0,key
assert out[4][:,:,0].abs().sum()==0
m.eval()
with torch.no_grad():
 full=m(t,v,mask_t=mask)[2];m.transport='none';common=m(t,v,mask_t=mask)[2]
 assert common.abs().sum()>0 and (full-common).abs().sum()>0
 assert torch.allclose(common[0,0],common[0,1])
 assert common[0,3:].abs().sum()==0
 assert m(t,torch.zeros_like(v),mask_t=mask)[2].abs().sum()==0
 m.ib='off';m.transport='uniform_partial';o=m(t,v,mask_t=mask);assert torch.isfinite(o[2]).all()
print('PASS gradients, CLS exclusion, padding, preserved common path, missing images, ablations')
