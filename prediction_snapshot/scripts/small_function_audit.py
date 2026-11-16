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
    rng = np.random.default_rng(42+cli.repeat)
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
    parser.add_argument('--kind',choices=['deletion','corruption'],required=True)
    parser.add_argument('--repeat',type=int,default=0)
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
    n = len(dataset)
    rng = np.random.default_rng(42+cli.repeat)
    # A fixed label-independent donor for every post, with a different image ID.
    donor = np.empty(n, dtype=int)
    image_ids = np.array([str(e.imgid) for e in examples])
    for i in range(n):
        eligible = np.flatnonzero(image_ids != image_ids[i])
        if not len(eligible): raise ValueError('No distinct donor image')
        donor[i] = rng.choice(eligible)
    (dest/'donors.json').write_text(json.dumps(donor.tolist()))
    rows=[dict(name='native',fraction=0.,gate='native',plan='native')]
    if cli.kind=='deletion':
        for fraction in [.1,.2,.3,.5]:
            for mode in ['high','low','random0','random1','random2']:
                rows.append(dict(name=f'delete_{mode}_{fraction}',fraction=fraction,gate=mode,plan='native'))
    else:
        for fraction in [.1,.25,.5,.75]:
            for gate in ['native','shuffle']:
                for plan in ['native','independent']:
                    rows.append(dict(name=f'noise_{fraction}_{gate}_{plan}',fraction=fraction,gate=gate,plan=plan))
    (dest/'protocol.json').write_text(json.dumps({'kind':cli.kind,'repeat':cli.repeat,'conditions':rows,'deletion':'Set selected gates to zero, preserve all slots and OT capacities, recompute cost and transport. Rank by original clean-image gate. No physical region removal.','corruption':'Replace non-CLS encoded patches with different-image patches. Random external patches are not guaranteed semantically irrelevant. Nested masks; three fixed donor seeds.','primary':'Entity CE increase and gold-class probability decrease relative to native; per-region gate and transported mass, normalized by region count. F1 secondary.','scope':'DEV inference diagnostics, not training ablation or proof of semantic grounding.'},indent=2))
    current={};clean_gate=None;offset=0
    def gate_hook(module,inputs,result):
        mode=current['gate']
        if mode=='native':return result
        x,mask=inputs[:2];g=module.current_gate.clone()
        for i in range(len(g)):
            ix=mask[i].bool().nonzero().flatten()
            if mode=='shuffle':
                gen=torch.Generator(device=g.device).manual_seed(4200+cli.repeat*10000+offset+i)
                g[i,ix]=g[i,ix[torch.randperm(len(ix),device=g.device,generator=gen)]]
            else:
                k=round(current['fraction']*len(ix))
                if mode.startswith('random'):
                    gen=torch.Generator(device=g.device).manual_seed(4200+int(mode[-1])*10000+offset+i)
                    order=torch.randperm(len(ix),device=g.device,generator=gen)
                else:order=torch.argsort(clean_gate[i,ix,0],descending=mode=='high')
                g[i,ix[order[:k]]]=0
                assert int((g[i,ix]==0).sum())>=k
        module.current_gate=g;module.last_gate=g.detach()
        return g*module.mu(x)*mask[...,None],result[1],result[2]
    def plan_hook(module,inputs,result):
        if current['plan']=='native':return result
        p,l=result;q=decorrelate_plan(p)
        assert torch.allclose(p.sum(1),q.sum(1),atol=1e-6) and torch.allclose(p.sum(2),q.sum(2),atol=1e-6)
        return q,l
    aligner.ib_v.register_forward_hook(gate_hook);aligner.solver.register_forward_hook(plan_hook)
    counts={r['name']:[] for r in rows};metrics={r['name']:[] for r in rows};region=[]
    with torch.inference_mode():
        for cpu in DataLoader(dataset,batch_size=8,collate_fn=collate_fn,shuffle=False):
            batch={k:v.to(pg.args.device) for k,v in cpu.items()};bs=len(batch['input_ids']);mask=batch['span_masks'].bool();gold=batch['span_labels']
            text,_,attention=model.span_encoding(**batch);_,visual=model.image_model(batch['image'])
            donor_visual=None
            if cli.kind=='corruption':
                dc=collate_fn([dataset[int(donor[i])] for i in range(offset,offset+bs)])
                _,donor_visual=model.image_model(dc['image'].to(pg.args.device))
            patch_count=visual.shape[1]-1
            ranks=torch.tensor(np.stack([np.argsort(np.random.default_rng(4200+cli.repeat*10000+i).permutation(patch_count)) for i in range(offset,offset+bs)]),device=visual.device)
            baseline=None
            for row in rows:
                current.update(row)
                corrupt=torch.zeros(bs,visual.shape[1],dtype=torch.bool,device=visual.device)
                image=visual
                if cli.kind=='corruption' and row['fraction']:
                    corrupt[:,1:]=ranks<round(row['fraction']*patch_count)
                    image=torch.where(corrupt[...,None],donor_visual,visual)
                out=aligner(text,image,mask_t=mask.float(),attn_t=attention);t,_,at=out[:3]
                logits=model.span_classifier(model.ln_text(model.fusion_text(torch.cat([t,at],-1))+at))
                if getattr(model,'_misc_logit_bias',None) is not None and model.id_misc_focal>=0:logits[:,:,model.id_misc_focal]+=float(model._misc_logit_bias)
                probs=model.softmax(logits);pred=probs.argmax(-1)
                if row['name']=='native':
                    baseline=pred.clone();clean_gate=aligner.ib_v.current_gate.clone()
                    if offset==0:
                        ref,_=model({**batch,'span_labels':None});assert torch.allclose(probs,ref[0],atol=2e-6)
                ce=torch.nn.functional.cross_entropy(logits.transpose(1,2),gold,reduction='none')
                gp=probs.gather(-1,gold.clamp_min(0)[...,None]).squeeze(-1)
                pp=probs.cpu()
                for i in range(bs):
                    cc,_=pg.span_f1_prune(cpu['span_word_idxes'][i:i+1],pp[i:i+1],cpu['span_labels'][i:i+1],cpu['span_masks'][i:i+1]);counts[row['name']].append(cc.numpy())
                    entity=mask[i]&(gold[i]!=0);valid=mask[i]
                    metrics[row['name']].append([float(ce[i][entity].sum()),float(gp[i][entity].sum()),int(entity.sum()),int((pred[i][valid]!=baseline[i][valid]).sum()),int(valid.sum())])
                if cli.kind=='corruption' and row['gate']=='native' and row['plan']=='native' and row['fraction']:
                    gates=aligner.ib_v.current_gate.squeeze(-1);mass=out[4].sum(1)
                    for i in range(bs):
                        if not visual[i].abs().sum() or not donor_visual[i].abs().sum():continue
                        bad=corrupt[i];good=~bad;good=good.clone();good[0]=False
                        region.append({'index':offset+i,'image_id':str(examples[offset+i].imgid),'donor_index':int(donor[offset+i]),'fraction':row['fraction'],'foreign_gate':float(gates[i][bad].mean()),'original_gate':float(gates[i][good].mean()),'foreign_mass_per_region':float(mass[i][bad].mean()),'original_mass_per_region':float(mass[i][good].mean()),'foreign_unused_capacity':float((1-mass[i][bad]*patch_count).mean()),'original_unused_capacity':float((1-mass[i][good]*patch_count).mean())})
            offset+=bs;(dest/'progress.json').write_text(json.dumps({'processed':offset,'total':n}));print(offset,flush=True)
            if cli.limit and offset>=cli.limit:break
    arrays={k:np.stack(v) for k,v in counts.items()};mm={k:np.array(v) for k,v in metrics.items()}
    np.savez_compressed(dest/'per_post_counts.npz',**arrays);np.savez_compressed(dest/'per_post_metrics.npz',**mm)
    scores={k:float(f1(v)) for k,v in arrays.items()}
    if not cli.limit:
        expected=max(map(float,re.findall(r'^span_f1 = ([0-9.]+)',(run/'eval_results.txt').read_text(),re.M)));assert abs(scores['native']-expected)<1.1e-5,(scores['native'],expected)
    aggregates={}
    for k,m in mm.items():
        z=m.sum(0);aggregates[k]={'entity_ce':float(z[0]/max(z[2],1)),'entity_gold_probability':float(z[1]/max(z[2],1)),'entities':int(z[2]),'changed_candidates':int(z[3]),'valid_candidates':int(z[4]),'f1':scores[k]}
    (dest/'region_records.json').write_text(json.dumps(region,indent=2))
    (dest/'results.json').write_text(json.dumps({'complete':not bool(cli.limit),'processed':offset,'kind':cli.kind,'repeat':cli.repeat,'conditions':aggregates},indent=2))
    print('DONE',cli.kind,cli.repeat)

if __name__=='__main__':main()
