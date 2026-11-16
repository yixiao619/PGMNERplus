from pathlib import Path
import json
import torch
import numpy as np
from run import solve

HERE=Path(__file__).resolve().parent
base=json.loads((HERE.parent/'results.json').read_text())
protocol=json.loads((HERE/'BD_protocol.json').read_text())
result={'B':{},'D':[]}
for name in ['balanced_full','balanced_no_polarization']:
    by={r['step']:r for r in json.loads((HERE.parent/f'{name}_probe_snapshot.json').read_text())}
    rows=[]
    for step in base['B']['matched_steps']:
        r=by[step]
        values={}
        for modality in ['text','image']:
            m=r[f'{modality}_gate_mean'];sd=r[f'{modality}_gate_std']
            values[modality]=sd**2/(m*(1-m))
            assert -.00001<=values[modality]<=1.00001
        rows.append(dict(step=step,**values))
    result['B'][name]=dict(trace=rows,last_five_mean={k:float(np.mean([r[k] for r in rows[-5:]])) for k in ['text','image']})
similarity=torch.eye(4,dtype=torch.float64)[:3]
for mass in protocol['D']['mass']:
    for eps in protocol['D']['eps']:
        for g in protocol['D']['gate']:
            for mode in protocol['D']['costs']:
                cost=1-similarity
                if mode=='retention_aware':
                    gv=torch.full((4,),.8,dtype=torch.float64);gv[0]=g
                    r=.8*gv[None]
                    cost=r*cost+2*(1-r)
                p,info=solve(cost,eps,mass)
                value=float(p[:,0].sum())
                lower=max(0,mass-.75)
                assert lower-1e-6<=value<=.25+1e-6
                result['D'].append(dict(mass=mass,eps=eps,gate=g,mode=mode,patch_mass=value,
                                        capacity_floor=lower,**info))
out=HERE/'BD_results.json'
if out.exists():raise SystemExit('Refuse overwrite')
out.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v['last_five_mean'] for k,v in result['B'].items()},indent=2))
print('D conditions',len(result['D']),'all converged',all(r['converged'] for r in result['D']))
