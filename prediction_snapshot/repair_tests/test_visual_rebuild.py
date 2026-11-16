import os
import torch
from models.repaired_alignment import RepairedAligner

def test_path():
 torch.manual_seed(42)
 m=RepairedAligner(12,8,16,n_iters=30,eps=.1,safe_fusion=True,cost_mode='gate_reliability')
 for ib in (m.ib_t,m.ib_v):
  ib.balance_mode='bernoulli'; ib.noise_mode='gated_posterior'
 t=torch.randn(2,5,12); v=torch.randn(2,7,8); mask=torch.tensor([[1,1,1,0,0],[1,1,1,1,1.]])
 out=m(t,v,mask_t=mask)
 loss=out[2].square().mean()+out[5]*.01+out[6]['explore_aux']+.01*out[6]['repair_polar']
 loss.backward()
 for name in ['context_t.weight','context_v.weight','visual_proj.weight','match_t.weight','cross_logit','ib_t.gate.weight','ib_v.gate.weight']:
  g=dict(m.named_parameters())[name].grad
  assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0,name
 assert out[4][0,3:].abs().sum()==0
 m.eval()
 with torch.no_grad():
  native=m(t,v,mask_t=mask)[2]
  wrong=m(t,v.flip(0),mask_t=mask)[2]
  assert (native-wrong).abs().max()>1e-4
  missing=m(t,torch.zeros_like(v),mask_t=mask)[2]
  assert missing.abs().sum()==0
 m.transport='none'
 assert m(t,v,mask_t=mask)[2].abs().sum()==0
 print('PASS: gradient, padding, image sensitivity, missing image, no-alignment checks')
if __name__=='__main__': test_path()
