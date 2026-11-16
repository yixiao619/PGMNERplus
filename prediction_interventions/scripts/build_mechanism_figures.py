from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SRC=Path('/data/jyx/PGMNER_small_function_archive_20261108');OUT=Path('/data/jyx/PGMNER_supplement_20261108');OUT.mkdir(exist_ok=True)
s=json.loads((SRC/'summary.json').read_text());d=np.load(SRC/'deletion/per_post_metrics.npz')
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
def ce(x):return x[...,0].sum(-1)/np.maximum(x[...,2].sum(-1),1)
def interval(a,b):
 ix=np.random.default_rng(42).integers(len(a),size=(1000,len(a)));q=ce(a[ix])-ce(b[ix]);return float(ce(a)-ce(b)),np.quantile(q,[.025,.975])
def save(fig,name):
 fig.tight_layout();fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
fs=[.1,.2,.3,.5];fig,ax=plt.subplots(1,2,figsize=(10,3.8))
for mode,label,color in [('high','High gate','#be4b48'),('low','Low gate','#2a788e'),('random','Random (3 orders)','#666666')]:
 vals=[];cis=[]
 for f in fs:
  a=np.mean([d[f'delete_random{i}_{f}'] for i in range(3)],0) if mode=='random' else d[f'delete_{mode}_{f}']
  v,c=interval(a,d['native']);vals.append(v);cis.append(c)
 vals=np.array(vals);cis=np.array(cis);x=np.array(fs)*100
 ax[0].plot(x,vals,'o-',label=label,color=color);ax[0].fill_between(x,cis[:,0],cis[:,1],alpha=.1,color=color)
for i in range(3):ax[0].plot(np.array(fs)*100,[ce(d[f'delete_random{i}_{f}'])-ce(d['native']) for f in fs],':',color='gray',alpha=.4,lw=.8)
ax[0].set(xlabel='Regions with gate set to zero (%)',ylabel='Change in entity cross-entropy',title='(a) Importance-ranking trend');ax[0].legend(fontsize=8);ax[0].axhline(0,color='gray',lw=.7)
fs2=[.1,.25,.5,.75];v=np.array([s['region_filtering'][str(f)]['original_minus_foreign_gate']['mean'] for f in fs2]);c=np.array([s['region_filtering'][str(f)]['original_minus_foreign_gate']['ci95'] for f in fs2]);ax[1].errorbar(np.array(fs2)*100,v,yerr=[v-c[:,0],c[:,1]-v],fmt='o-',capsize=4,color='#2a788e');ax[1].axhline(0,color='gray',lw=.7);ax[1].set(xlabel='Foreign encoded regions (%)',ylabel='Original minus foreign mean gate',title='(b) Gate response to interference')
save(fig,'fig_ppib_function')
fig,ax=plt.subplots(figsize=(5.7,3.8));v=np.array([s['corruption'][str(f)]['independent_minus_native']['delta_entity_ce'] for f in fs2]);c=np.array([s['corruption'][str(f)]['independent_minus_native']['ci95'] for f in fs2]);ax.errorbar(np.array(fs2)*100,v,yerr=[v-c[:,0],c[:,1]-v],fmt='o-',capsize=4,label='Mean and post-bootstrap 95% CI',color='#2a788e')
for i in range(3):
 r=json.loads((SRC/f'corruption{i}/results.json').read_text())['conditions'];ax.plot(np.array(fs2)*100,[r[f'noise_{f}_native_independent']['entity_ce']-r[f'noise_{f}_native_native']['entity_ce'] for f in fs2],':',alpha=.6,label=f'Donor pairing {i+1}')
ax.axhline(0,color='gray',lw=.7);ax.set(xlabel='Foreign encoded regions (%)',ylabel='CE(independent plan) - CE(original plan)',title='GOTA: utility of specific matching');ax.legend(fontsize=8);save(fig,'fig_gota_matching')
fig,ax=plt.subplots(figsize=(5.7,3.8));v=np.array([s['region_filtering'][str(f)]['foreign_minus_original_unused_capacity']['mean'] for f in fs2])*100;c=np.array([s['region_filtering'][str(f)]['foreign_minus_original_unused_capacity']['ci95'] for f in fs2])*100;ax.errorbar(np.array(fs2)*100,v,yerr=[v-c[:,0],c[:,1]-v],fmt='o-',capsize=4,color='#39568c');ax.axhline(0,color='gray',lw=.7);ax.set(xlabel='Foreign encoded regions (%)',ylabel='Foreign minus original unused capacity (pp)',title='Partial transport: differential rejection');save(fig,'fig_partial_rejection')
# Deterministic case ranking using exact pruned entity errors.
a=np.load('/data/jyx/PGMNER_stage_archive_20261108/experiment/runs/selected_reduced/audit/per_post_counts.npz');b=np.load(SRC/'corruption0/per_post_counts.npz');rows=[];chosen=[]
def err(z):return z[:,1]+z[:,2]-2*z[:,0]
for kind,x,y in [('gate',a['n0_shuffle_both_native'],a['n0_native_native']),('matching',a['n0_native_independent'],a['n0_native_native']),('failure',b['noise_0.75_native_native'],b['native'])]:
 delta=err(x)-err(y);ix=sorted(np.where(delta>0)[0],key=lambda i:(-delta[i],i))
 for rank,i in enumerate(ix):rows.append({'kind':kind,'rank':rank+1,'post_index':int(i),'extra_FP_FN':float(delta[i])})
 if ix:chosen.append({'kind':kind,'post_index':int(ix[0]),'extra_FP_FN':float(delta[ix[0]])})
with (OUT/'case_ranking.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['kind','rank','post_index','extra_FP_FN']);w.writeheader();w.writerows(rows)
(OUT/'case_selection.json').write_text(json.dumps({'rule':'Largest increase in pruned FP+FN; ties lowest dataset index. Positive-gate, positive-matching, and75%-interference failure. Illustrative selection, not representative sampling.','cases':chosen},indent=2))
print(chosen)
