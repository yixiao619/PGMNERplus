"""Three prespecified DEV interventions. Never train, tune thresholds, or read TEST."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

def decorrelate_plan(plan):
    """Destroy assignment while preserving BOTH marginals and total mass."""
    mass = plan.sum((1, 2), keepdim=True)
    return plan.sum(2, keepdim=True) * plan.sum(1, keepdim=True) / mass.clamp_min(1e-30)

def change_gate(gate, mask, mode, seed):
    if mode in ('keep_high','keep_low','keep_random'):
        out=torch.zeros_like(gate)
        for i in range(len(gate)):
            ix=mask[i].bool().nonzero().flatten();k=max(1,min(len(ix),round(float(gate[i,ix].mean())*len(ix))))
            values=gate[i,ix,0]
            if mode=='keep_random':
                gen=torch.Generator(device=gate.device).manual_seed(seed+i);order=torch.randperm(len(ix),device=gate.device,generator=gen)
            else:order=torch.argsort(values,descending=mode=='keep_high')
            out[i,ix[order[:k]]]=1
            assert int(out[i].sum())==k
        return out
    out = gate.clone()
    gen = torch.Generator(device=gate.device).manual_seed(seed)
    for i in range(len(gate)):
        ix = mask[i].bool().nonzero().flatten()
        if mode == 'mean':
            out[i, ix] = gate[i, ix].mean()
        else:
            perm = torch.randperm(len(ix), device=gate.device, generator=gen)
            out[i, ix] = gate[i, ix[perm]]
        assert torch.allclose(out[i, ix].mean(), gate[i, ix].mean(), atol=1e-6)
    return out

def conditions():
    rows = []
    for noise in [0., .25, .5, .75]:
        for gate in ['native', 'shuffle_both']:
            for plan in ['native', 'independent']:
                rows.append(dict(name=f'n{noise:g}_{gate}_{plan}', noise=noise,
                                 gate=gate, plan=plan, wrong_image=False))
    for gate in ['shuffle_text', 'shuffle_image', 'mean']:
        rows.append(dict(name=gate, noise=0., gate=gate, plan='native', wrong_image=False))
    rows.append(dict(name='wrong_image', noise=0., gate='native', plan='native', wrong_image=True))
    return rows

def f1(counts):
    summed = counts.sum(axis=-2)
    return 2*summed[..., 0] / np.maximum(summed[..., 1]+summed[..., 2], 1)

def paired_estimates(counts):
    keys = list(counts)
    n = len(counts[keys[0]])
    rng = np.random.default_rng(42)
    samples = rng.integers(n, size=(1000, n))
    definitions = {
        'E1_gate_assignment': [('n0_native_native', 1), ('n0_shuffle_both_native', -1)],
        'E2_transport_assignment': [('n0_native_native', 1), ('n0_native_independent', -1)],
        'E2_correct_image': [('n0_native_native', 1), ('wrong_image', -1)],
        'E3_interaction_at_50pct_noise': [('n0.5_native_native', 1),
            ('n0.5_shuffle_both_native', -1), ('n0.5_native_independent', -1),
            ('n0.5_shuffle_both_independent', 1)],
    }
    output = {}
    for name, terms in definitions.items():
        observed = sum(sign*float(f1(counts[key])) for key, sign in terms)
        boot = sum(sign*f1(counts[key][samples]) for key, sign in terms)
        output[name] = dict(delta_f1=observed, ci95=np.quantile(boot, [.025, .975]).tolist())
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--limit', type=int, default=0, help='Smoke check only; never a full result')
    cli = parser.parse_args()
    run, dest = Path(cli.run).resolve(), Path(cli.output).resolve()
    os.environ['PGMNER_EXPERIMENT']=(run/'experiment.json').read_text()
    dest.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    cmd = json.loads((run/'command.json').read_text())
    for key, value in {'--gpu_id': str(cli.gpu), '--do_train': 'false', '--do_eval': 'false',
                       '--do_predict': 'false', '--output_dir': str(dest)}.items():
        cmd[cmd.index(key)+1] = value
    sys.argv = cmd[1:]
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('pgm_audit', ROOT/'main-mner-robert.py')
    pg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pg)
    labels = pg.get_labels(pg.args.labels)
    cfg = pg.AutoConfig.from_pretrained(pg.args.model_name_or_path)
    tok = pg.RobertaTokenizer.from_pretrained(pg.args.tokenizer_name)
    model = pg.BertSpanNerBN(cfg, args=pg.args, num_labels=len(labels))
    state = torch.load(run/'best_model.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(state.get('state_dict', state), strict=True)
    del state
    model.to(pg.args.device).eval()
    model.set_partial_ot_warmup(False)
    aligner = model.semantic_ot_aligner
    examples = pg.load_examples(pg.args.data_dir, mode='dev', tokenizer=tok,
        max_seq_len=pg.args.max_seq_len, aux_dropout=0., vllm_desc_dropout=0.)
    from utils.datasets import SpanNerDataset, collate_fn
    from torch.utils.data import DataLoader
    dataset = SpanNerDataset(examples, args=pg.args, tokenizer=tok, labels=labels, dev=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import textwrap
    cases=json.loads((dest.parent/'case_selection.json').read_text())['cases']
    ref=np.load('/data/jyx/PGMNER_stage_archive_20261108/experiment/runs/selected_reduced/audit/per_post_counts.npz')
    noise_ref=np.load('/data/jyx/PGMNER_small_function_archive_20261108/corruption0/per_post_counts.npz')
    donors=json.loads(Path('/data/jyx/PGMNER_small_function_archive_20261108/corruption0/donors.json').read_text())
    current={'mode':'native','batch':0}
    def gh(m,args,result):
        if current['mode']!='gate':return result
        x,mask=args[:2];g=change_gate(m.current_gate,mask,'shuffle',42+current['batch']*100+1)
        m.current_gate=g;m.last_gate=g.detach();return g*m.mu(x)*mask[...,None],result[1],result[2]
    def ph(m,args,result):
        return (decorrelate_plan(result[0]),result[1]) if current['mode']=='matching' else result
    aligner.ib_v.register_forward_hook(gh);aligner.solver.register_forward_hook(ph)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42});fig,axs=plt.subplots(len(cases),3,figsize=(11,4*len(cases)));records=[]
    with torch.inference_mode():
        for row,case in enumerate(cases):
            idx=case['post_index'];start=idx//8*8;end=min(start+8,len(dataset));pos=idx-start
            cpu=collate_fn([dataset[i] for i in range(start,end)]);batch={k:v.to(pg.args.device) for k,v in cpu.items()}
            text,_,attention=model.span_encoding(**batch);_,visual=model.image_model(batch['image']);data={};foreign=torch.zeros(visual.shape[:2],dtype=torch.bool,device=visual.device)
            altered_image=visual
            if case['kind']=='failure':
                dc=collate_fn([dataset[int(donors[i])] for i in range(start,end)]);_,dv=model.image_model(dc['image'].to(pg.args.device));pc=visual.shape[1]-1
                ranks=torch.tensor(np.stack([np.argsort(np.random.default_rng(4200+i).permutation(pc)) for i in range(start,end)]),device=visual.device)
                foreign[:,1:]=ranks<round(.75*pc);altered_image=torch.where(foreign[...,None],dv,visual)
            for mode in ['native',case['kind']]:
                current.update(mode=mode,batch=idx//8)
                out=aligner(text,altered_image if mode=='failure' else visual,mask_t=batch['span_masks'].float(),attn_t=attention);t,_,at=out[:3]
                logits=model.span_classifier(model.ln_text(model.fusion_text(torch.cat([t,at],-1))+at))
                if getattr(model,'_misc_logit_bias',None) is not None and model.id_misc_focal>=0:logits[:,:,model.id_misc_focal]+=float(model._misc_logit_bias)
                probs=model.softmax(logits);cc,pruned=pg.span_f1_prune(cpu['span_word_idxes'][pos:pos+1],probs[pos:pos+1].cpu(),cpu['span_labels'][pos:pos+1],cpu['span_masks'][pos:pos+1])
                expected=ref['n0_native_native'][idx] if mode=='native' else (ref['n0_shuffle_both_native'][idx] if mode=='gate' else ref['n0_native_independent'][idx] if mode=='matching' else noise_ref['noise_0.75_native_native'][idx])
                assert np.array_equal(cc.numpy(),expected),(idx,mode,cc,expected)
                data[mode]={'counts':cc.numpy(),'pruned':pruned[0].numpy(),'prob':probs[pos].cpu().numpy(),'gate':aligner.ib_v.current_gate[pos,1:,0].cpu().numpy(),'plan':out[4][pos,:,1:].cpu().numpy()}
            native=data['native'];altered=data[case['kind']];gold=cpu['span_labels'][pos].numpy();mask=cpu['span_masks'][pos].bool().numpy()
            useful=np.where(mask&(gold!=0)&(native['pruned']==gold)&(altered['pruned']!=gold))[0]
            if not len(useful):useful=np.where(mask&(native['pruned']!=altered['pruned']))[0]
            assert len(useful)>0;target=int(useful[0]);ws,we=map(int,cpu['span_word_idxes'][pos,target].tolist());words=examples[idx].words;span=' '.join(words[ws:we+1]);state=altered if case['kind']=='failure' else native
            pic=(batch['image'][pos].cpu().numpy().transpose(1,2,0)*np.array([.26862954,.26130258,.27577711])+np.array([.48145466,.4578275,.40821073])).clip(0,1)
            axs[row,0].imshow(pic);axs[row,0].axis('off');axs[row,0].set_title(f"{case['kind'].capitalize()} example | post {idx}\n"+textwrap.fill(span,35))
            im=axs[row,1].imshow(state['gate'].reshape(14,14),vmin=0,vmax=1,cmap='viridis');axs[row,1].set_title('PPIB gate'+(' (75% interference)' if case['kind']=='failure' else ' (native)'));fig.colorbar(im,ax=axs[row,1],fraction=.046,pad=.04)
            capacity=1/int(mask.sum());mass=state['plan'][target]/capacity
            im=axs[row,2].imshow(mass.reshape(14,14),cmap='magma',vmin=0);axs[row,2].set_title('GOTA mass / span capacity\n(no per-map sum normalization)');fig.colorbar(im,ax=axs[row,2],fraction=.046,pad=.04)
            for ax in axs[row,1:]:ax.set_xticks([]);ax.set_yticks([])
            if case['kind']=='failure':
                from matplotlib.patches import Rectangle
                for j in np.where(foreign[pos,1:].cpu().numpy())[0]:
                    for ax in axs[row,1:]:ax.add_patch(Rectangle((j%14-.5,j//14-.5),1,1,fill=False,edgecolor='red',lw=.35,alpha=.65))
            note=f"Gold: {labels[int(gold[target])]} | Native: {labels[int(native['pruned'][target])]} | Intervention: {labels[int(altered['pruned'][target])]}"
            axs[row,0].text(0,-.06,textwrap.fill(note,55),transform=axs[row,0].transAxes,va='top',fontsize=8)
            record={**case,'image_id':str(examples[idx].imgid),'words':words,'target_span':[ws,we],'target_text':span,'gold':labels[int(gold[target])],'native_prediction':labels[int(native['pruned'][target])],'intervention_prediction':labels[int(altered['pruned'][target])],'native_counts':native['counts'].tolist(),'intervention_counts':altered['counts'].tolist(),'donor_index':int(donors[idx]) if case['kind']=='failure' else None,'note':'Maps show patch weights, not annotated localization. Failure uses encoded-feature substitution; displayed photo is original crop, red cells mark substituted feature positions.'}
            records.append(record)
            np.savez_compressed(dest/f'case_{idx}_arrays.npz',native_gate=native['gate'],intervention_gate=altered['gate'],native_plan=native['plan'],intervention_plan=altered['plan'],foreign_mask=foreign[pos,1:].cpu().numpy())
    fig.tight_layout(h_pad=3);fig.savefig(dest/'fig_module_cases.pdf',bbox_inches='tight');fig.savefig(dest/'fig_module_cases.png',dpi=180,bbox_inches='tight')
    (dest/'case_records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));print(json.dumps(records,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
