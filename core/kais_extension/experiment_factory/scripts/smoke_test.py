#!/usr/bin/env python3
from __future__ import annotations
import json,os,sys
from pathlib import Path
import torch
ROOT=Path("/home/jyx/yixiao/MINER-My");sys.path.insert(0,str(ROOT));F=Path(os.environ.get("FACTORY_ROOT",ROOT/"kais_extension/experiment_factory"))
def main():
 cfg=json.loads((F/"configs/routes.json").read_text());results=[];B,N,M,D=2,5,7,1024;zt=torch.randn(B,N,D,requires_grad=True);zv=torch.randn(B,M,D,requires_grad=True);C=.5*(1-torch.einsum("bnd,bmd->bnm",torch.nn.functional.normalize(zt,dim=-1),torch.nn.functional.normalize(zv,dim=-1)));mt=torch.ones(B,N);mv=torch.ones(B,M)
 from kais_extension.experiment_factory.models.ot_backends import AdaptiveMassSinkhorn,FixedActualMassSinkhorn,ReliabilityWeightedSinkhorn
 for name,mod in [("G2_ADAPTIVE",AdaptiveMassSinkhorn(eps=.1,dust_cost=1,n_iters=5)),("G2_FIXED",FixedActualMassSinkhorn(eps=.1,dust_cost=1,n_iters=5)),("G3_RELIABILITY",ReliabilityWeightedSinkhorn(eps=.1,dust_cost=1,n_iters=5,dim=D))]:
  T,L=mod(zt,zv,C_pre=C,mask_t=mt,mask_v=mv);L.backward(retain_graph=True);params=[p for p in mod.parameters() if p.requires_grad];connected=any(p.grad is not None and torch.isfinite(p.grad).all() for p in params);results.append({"id":name,"shape":list(T.shape),"finite":bool(torch.isfinite(T).all() and torch.isfinite(L)),"parameter_gradient":connected,"stats":mod.last_factory_stats})
 a=AdaptiveMassSinkhorn(eps=.1,dust_cost=1,n_iters=10);f=FixedActualMassSinkhorn(eps=.1,dust_cost=1,n_iters=10);r=ReliabilityWeightedSinkhorn(eps=.1,dust_cost=1,n_iters=10,dim=D);Ta,_=a(zt,zv,C_pre=C,mask_t=mt,mask_v=mv);Tf,_=f(zt,zv,C_pre=C,mask_t=mt,mask_v=mv);Tr,_=r(zt,zv,C_pre=C,mask_t=mt,mask_v=mv);results.append({"id":"INITIAL_EQUIVALENCE","adaptive_vs_fixed_max_abs":float((Ta-Tf).abs().max()),"reliability_vs_fixed_max_abs":float((Tr-Tf).abs().max()),"pass":float((Ta-Tf).abs().max())<1e-6 and float((Tr-Tf).abs().max())<1e-6})
 known={j["route"] for j in cfg["jobs"]};results.append({"id":"CONFIG","routes":sorted(known),"unique_ids":len({j['id'] for j in cfg['jobs']})==len(cfg['jobs']),"full_training_disabled":not cfg["stage3"]["enable_full_training"]})
 (F/"results/synthetic_smoke.json").write_text(json.dumps(results,indent=2)+"\n");print(json.dumps(results,indent=2));assert all(x.get("finite",True) for x in results) and results[-2]["pass"]
if __name__=="__main__":main()
