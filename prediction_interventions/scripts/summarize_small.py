import json,csv,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--base',required=True);cli=p.parse_args();base=Path(cli.base)
def load(name):return np.load(base/name/'per_post_metrics.npz')
def ce(a):return a[...,0].sum(-1)/np.maximum(a[...,2].sum(-1),1)
def paired(a,b):
 rng=np.random.default_rng(42);ix=rng.integers(len(a),size=(1000,len(a)))
 boot=ce(a[ix])-ce(b[ix]);return {'delta_entity_ce':float(ce(a)-ce(b)),'ci95':np.quantile(boot,[.025,.975]).tolist()}
d=load('deletion');noise=[load('corruption'+str(i)) for i in range(3)];summary={'deletion':{},'corruption':{},'region_filtering':{},'note':'Exploratory fixed-checkpoint DEV diagnostics. CI resamples posts after averaging fixed repeats; no training-seed uncertainty. Positive deletion high-minus-random CE supports importance ranking. Positive shuffled-minus-native CE supports gate utility. Foreign patches need not be semantically irrelevant.'}
fractions=[.1,.2,.3,.5];lines={k:[] for k in ['high','low','random']}
for f in fractions:
 rand=np.mean([d[f'delete_random{i}_{f}'] for i in range(3)],axis=0)
 summary['deletion'][str(f)]={'high_minus_random':paired(d[f'delete_high_{f}'],rand),'high_minus_low':paired(d[f'delete_high_{f}'],d[f'delete_low_{f}'])}
 for k,a in [('high',d[f'delete_high_{f}']),('low',d[f'delete_low_{f}']),('random',rand)]:lines[k].append(float(ce(a)-ce(d['native'])))
for f in [.1,.25,.5,.75]:
 means={k:np.mean([x[f'noise_{f}_{k}'] for x in noise],axis=0) for k in ['native_native','shuffle_native','native_independent','shuffle_independent']}
 summary['corruption'][str(f)]={'shuffle_minus_native':paired(means['shuffle_native'],means['native_native']),'independent_minus_native':paired(means['native_independent'],means['native_native']),'both_minus_native':paired(means['shuffle_independent'],means['native_native'])}
 # Interaction = CE(both) - CE(shuffle) - CE(independent) + CE(native).
 a=means['shuffle_independent'].copy();b=means['shuffle_native'].copy();a[:,0]-=means['native_independent'][:,0];b[:,0]-=means['native_native'][:,0]
 summary['corruption'][str(f)]['interaction']=paired(a,b)
 records=[json.loads((base/('corruption'+str(i))/'region_records.json').read_text()) for i in range(3)]
 perpost={}
 for rr in records:
  for r in rr:
   if r['fraction']!=f:continue
   perpost.setdefault(r['index'],[]).append([r['original_gate']-r['foreign_gate'],r['original_mass_per_region']-r['foreign_mass_per_region'],r['foreign_unused_capacity']-r['original_unused_capacity']])
 arr=np.array([np.mean(v,axis=0) for v in perpost.values()]);rng=np.random.default_rng(42);ix=rng.integers(len(arr),size=(1000,len(arr)));boot=arr[ix].mean(1)
 summary['region_filtering'][str(f)]={k:{'mean':float(arr[:,j].mean()),'ci95':np.quantile(boot[:,j],[.025,.975]).tolist()} for j,k in enumerate(['original_minus_foreign_gate','original_minus_foreign_mass_per_region','foreign_minus_original_unused_capacity'])}
 summary['region_filtering'][str(f)]['posts']=len(arr)
(base/'summary.json').write_text(json.dumps(summary,indent=2))
fig,axs=plt.subplots(1,2,figsize=(10,4))
for k,v in lines.items():axs[0].plot(np.array(fractions)*100,v,marker='o',label=k)
axs[0].set(xlabel='Regions with gates set to zero (%)',ylabel='Entity CE increase',title='PPIB deletion sensitivity');axs[0].legend();axs[0].axhline(0,color='gray',lw=.5)
fs=[.1,.25,.5,.75]
for label,key in [('gate permutation','shuffle_minus_native'),('independent matching','independent_minus_native'),('both','both_minus_native')]:axs[1].plot(np.array(fs)*100,[summary['corruption'][str(f)][key]['delta_entity_ce'] for f in fs],marker='o',label=label)
axs[1].set(xlabel='Foreign encoded regions (%)',ylabel='CE(intervention) - CE(native)',title='Gate / matching utility under interference');axs[1].legend();axs[1].axhline(0,color='gray',lw=.5)
fig.tight_layout();fig.savefig(base/'functional_curves.pdf');fig.savefig(base/'functional_curves.png',dpi=160)
rows=[]
for name in ['deletion','corruption0','corruption1','corruption2']:
 r=json.loads((base/name/'results.json').read_text())
 for condition,vals in r['conditions'].items():rows.append({'experiment':name,'condition':condition,**vals})
with (base/'all_conditions.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2))
