#!/usr/bin/env python3
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/jyx/yixiao/MINER-My'); EXP=ROOT/'kais_extension/experiment16c_ppib_audit';RES=EXP/'results';FIG=EXP/'figures';FIG.mkdir(parents=True,exist_ok=True)
def read(n): return list(csv.DictReader((RES/n).open()))
def write(n,rows):
 with (RES/n).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

audit=[
 {'configuration':'Full PGMNER','actual_change':'PP-IB enabled; beta_polar=0.01; balance hinge lambda=1; GOT-A enabled','archival_P':'0.8969','archival_R':'0.9163','archival_F1':'0.9065','seed':'42','checkpoint_available':'YES','evidence':'paper-full checkpoint + retained train/eval logs','comparability':'Reference; historical checkpoint selected by test_span_f1'},
 {'configuration':'w/o PP-IB','actual_change':'run_ablation.sh removes --use_pp_ib, --beta_polar, --delta_minus, --delta_plus','archival_P':'0.8822','archival_R':'0.9066','archival_F1':'0.8942','seed':'123','checkpoint_available':'NO','evidence':'archived summary and raw-max TSV; deleted-run manifest','comparability':'NOT CONTROLLED: different seed, 30 epochs/patience5 vs full 40/patience0, test-based gating'},
 {'configuration':'w/o Balanced Polarization','actual_change':'ONLY --pp_ib_lambda_balance 0; g(1-g) polarization remains active','archival_P':'0.8908','archival_R':'0.9081','archival_F1':'0.8993','seed':'456 (best archived)','checkpoint_available':'NO','evidence':'ablation_matrix.tsv + archived raw-max TSV','comparability':'Not a true w/o-polarization-loss run; different seed/budget; test-based selection'},
 {'configuration':'w/o polarization loss','actual_change':'Would require beta_polar=0; no matching retained run identified','archival_P':'','archival_R':'','archival_F1':'','seed':'','checkpoint_available':'NO','evidence':'UNAVAILABLE','comparability':'Cannot verify'},
 {'configuration':'w/o IB KL','actual_change':'Would require ib_kl_weight=0 or removal of loss_ib_tv; no matching retained run identified','archival_P':'','archival_R':'','archival_F1':'','seed':'','checkpoint_available':'NO','evidence':'UNAVAILABLE','comparability':'Cannot verify'},
 {'configuration':'w/o Numerical Stability','actual_change':'polar warmup=0, gate init std=0, logvar clamp=0 (compound change)','archival_P':'0.8848','archival_R':'0.9073','archival_F1':'0.8959','seed':'456 (best archived)','checkpoint_available':'NO','evidence':'ablation_matrix.tsv + archived raw-max TSV','comparability':'Compound ablation, not w/o IB or w/o polarization'},
]
write('ablation_resource_audit.csv',audit)

rep=read('representation_robustness.csv'); grad=read('gradient_contribution_summary.csv')
conds=['mild_blur','moderate_blur','random_image']; names=['Mild blur','Moderate blur','Random image']; colors=['#4C78A8','#F58518','#E45756']
def val(c,r,k): return float(next(x for x in rep if x['condition']==c and x['representation']==r)[k])
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'figure.dpi':150})
fig,ax=plt.subplots(figsize=(7.2,4));x=np.arange(3);w=.36
ax.bar(x-w/2,[val(c,'visual_fixed_gate_0.5','mean_normalized_l2') for c in conds],w,label='Fixed gate = 0.5',color='#4C78A8')
ax.bar(x+w/2,[val(c,'visual_gated','mean_normalized_l2') for c in conds],w,label='Learned gate',color='#E45756')
ax.set_xticks(x,names);ax.set_ylabel('Normalized L2 drift');ax.set_title('PP-IB latent stability: learned vs fixed scaling');ax.legend(frameon=False);fig.tight_layout();fig.savefig(FIG/'fig16c_a_ppib_stability.pdf');fig.savefig(FIG/'fig16c_a_ppib_stability.png',dpi=300);plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,4)); reps=['visual_pre','visual_mu','visual_gated','fused_classifier_input'];labs=['Visual global input','PP-IB mean $\\mu$','Gated sampled latent','Classifier input'];x=np.arange(len(reps));w=.25
for i,(c,n,col) in enumerate(zip(conds,names,colors)):ax.bar(x+(i-1)*w,[val(c,r,'mean_normalized_l2') for r in reps],w,label=n,color=col)
ax.set_xticks(x,labs,rotation=12,ha='right');ax.set_yscale('log');ax.set_ylabel('Normalized L2 drift (log scale)');ax.set_title('Perturbation drift across representations');ax.legend(frameon=False);fig.tight_layout();fig.savefig(FIG/'fig16c_b_representation_drift.pdf');fig.savefig(FIG/'fig16c_b_representation_drift.png',dpi=300);plt.close(fig)
losses=['classification','supervised_contrastive','ib_kl','polarization','got'];groups=['text_gate','text_mu','text_logvar','visual_gate','visual_mu','visual_logvar'];A=np.zeros((len(losses),len(groups)))
for i,l in enumerate(losses):
 for j,g in enumerate(groups): A[i,j]=float(next(r for r in grad if r['loss_component']==l and r['parameter_group']==g)['grad_l2_mean'])
fig,ax=plt.subplots(figsize=(9,4));im=ax.imshow(np.log10(A+1e-12),cmap='viridis',vmin=-12,vmax=-2);ax.set_xticks(range(len(groups)),[x.replace('_','\n') for x in groups]);ax.set_yticks(range(len(losses)),losses);ax.set_title('Gradient routes into PP-IB parameters');cb=fig.colorbar(im,ax=ax);cb.set_label('$\\log_{10}(||g||_2 + 10^{-12})$');fig.tight_layout();fig.savefig(FIG/'fig16c_c_gradient_routes.pdf');fig.savefig(FIG/'fig16c_c_gradient_routes.png',dpi=300);plt.close(fig)
summary={'clean_f1':val('clean','fused_classifier_input','f1'),'condition_f1':{c:val(c,'fused_classifier_input','f1') for c in conds},'fixed_vs_learned':{c:{'fixed_l2':val(c,'visual_fixed_gate_0.5','mean_normalized_l2'),'learned_l2':val(c,'visual_gated','mean_normalized_l2'),'learned_minus_fixed':val(c,'visual_gated','mean_normalized_l2')-val(c,'visual_fixed_gate_0.5','mean_normalized_l2')} for c in conds},'archival_ablation_caveat':'w/o PP-IB is seed123 and a different 30-epoch/patience5/test-gated protocol; w/o Balanced Polarization only removes the balance hinge. No retained checkpoints.','conclusion':'No evidence that learned gate improves robustness over fixed 0.5 scaling. IB KL trains mu/logvar; polarization alone trains gate; MNER and GOT losses have no gradient route to PP-IB module parameters.'}
(RES/'experiment16c_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
if __name__=='__main__': pass
