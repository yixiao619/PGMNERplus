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
    assert aligner.explore==json.loads((run/'experiment.json').read_text()), 'Checkpoint configuration mismatch'
    examples = pg.load_examples(pg.args.data_dir, mode='dev', tokenizer=tok,
        max_seq_len=pg.args.max_seq_len, aux_dropout=0., vllm_desc_dropout=0.)
    from utils.datasets import SpanNerDataset, collate_fn
    from torch.utils.data import DataLoader
    dataset = SpanNerDataset(examples, args=pg.args, tokenizer=tok, labels=labels, dev=True)
    n = len(dataset)
    rng = np.random.default_rng(42)
    # A fixed label-independent donor for every post, with a different image ID.
    donor = np.empty(n, dtype=int)
    image_ids = np.array([str(e.imgid) for e in examples])
    for i in range(n):
        eligible = np.flatnonzero(image_ids != image_ids[i])
        if not len(eligible): raise ValueError('No distinct donor image')
        donor[i] = rng.choice(eligible)
    (dest/'donors.json').write_text(json.dumps(donor.tolist()))
    names=['native','wrong_common','wrong_local','wrong_both','zero_common','zero_local','zero_both']
    counts={k:[] for k in names};posts=[];offset=0
    with torch.inference_mode():
        for cpu in DataLoader(dataset,batch_size=8,collate_fn=collate_fn,shuffle=False):
            batch={k:v.to(pg.args.device) for k,v in cpu.items()};bs=len(batch['input_ids']);mask=batch['span_masks'].bool()
            text,_,attention=model.span_encoding(**batch);_,visual=model.image_model(batch['image'])
            dc=collate_fn([dataset[int(donor[i])] for i in range(offset,offset+bs)])
            _,dv=model.image_model(dc['image'].to(pg.args.device))
            parts=[]
            for vv in [visual,dv]:
                out=aligner(text,vv,mask_t=mask.float(),attn_t=attention)
                t,_,at=out[:3]
                local=aligner.local_fusion_coefficient*aligner.local_aux_features*mask[...,None]*vv.abs().sum((1,2)).gt(0)[:,None,None]
                common=float(aligner.explore.get('common_coefficient',.1))*aligner.common_features*mask[...,None]*vv.abs().sum((1,2)).gt(0)[:,None,None]
                assert torch.equal(common+local,at),'Branch decomposition must preserve native output exactly'
                parts.append((common,local))
            c,l=parts[0];wc,wl=parts[1]
            variants=dict(native=c+l,wrong_common=wc+l,wrong_local=c+wl,wrong_both=wc+wl,zero_common=l,zero_local=c,zero_both=torch.zeros_like(c))
            predictions={};losses={}
            for mode,at in variants.items():
                logits=model.span_classifier(model.ln_text(model.fusion_text(torch.cat([t,at],-1))+at))
                if getattr(model,'_misc_logit_bias',None) is not None and model.id_misc_focal>=0:logits[:,:,model.id_misc_focal]+=float(model._misc_logit_bias)
                probs=model.softmax(logits)
                if mode=='native' and offset==0:
                    ref,_=model({**batch,'span_labels':None});assert torch.allclose(probs,ref[0],atol=2e-6)
                predictions[mode]=probs.argmax(-1).cpu()
                ce=torch.nn.functional.cross_entropy(logits.transpose(1,2),batch['span_labels'],reduction='none')
                losses[mode]=[(float(ce[i][mask[i]].mean()),float(ce[i][mask[i]&(batch['span_labels'][i]!=0)].mean()) if (mask[i]&(batch['span_labels'][i]!=0)).any() else None) for i in range(bs)]
                pp=probs.cpu()
                for i in range(bs):
                    cc,_=pg.span_f1_prune(cpu['span_word_idxes'][i:i+1],pp[i:i+1],cpu['span_labels'][i:i+1],cpu['span_masks'][i:i+1]);counts[mode].append(cc.numpy())
            for i in range(bs):
                ix=offset+i;changes=[]
                for j in cpu['span_masks'][i].bool().nonzero().flatten().tolist():
                    pred={k:int(v[i,j]) for k,v in predictions.items()}
                    if len(set(pred.values()))>1:
                        span=cpu['span_word_idxes'][i][j]
                        changes.append({'candidate_index':j,'span':span.tolist() if hasattr(span,'tolist') else str(span),'gold':int(cpu['span_labels'][i,j]),'predictions':pred})
                posts.append({'index':ix,'words':examples[ix].words,'bio_labels':examples[ix].bio_labels,'image_id':str(examples[ix].imgid),'donor_index':int(donor[ix]),'donor_image_id':str(examples[int(donor[ix])].imgid),'readable':bool(visual[i].abs().sum()),'donor_readable':bool(dv[i].abs().sum()),'ce':{k:v[i] for k,v in losses.items()},'changes':changes})
            offset+=bs;(dest/'progress.json').write_text(json.dumps({'processed':offset,'total':n}));print(offset,flush=True)
            if cli.limit and offset>=cli.limit:break
    arrays={k:np.stack(v) for k,v in counts.items()};np.savez_compressed(dest/'per_post_counts.npz',**arrays)
    scores={k:float(f1(v)) for k,v in arrays.items()}
    if not cli.limit:
        expected=max(map(float,re.findall(r'^span_f1 = ([0-9.]+)',(run/'eval_results.txt').read_text(),re.M)));assert abs(scores['native']-expected)<1.1e-5
    rng=np.random.default_rng(42);samples=rng.integers(offset,size=(1000,offset));paired={}
    for k in names[1:]:
        boot=f1(arrays['native'][samples])-f1(arrays[k][samples]);paired[k]={'native_minus_intervention':scores['native']-scores[k],'ci95':np.quantile(boot,[.025,.975]).tolist()}
    (dest/'samples.json').write_text(json.dumps({'labels':labels,'posts':posts},indent=2))
    (dest/'results.json').write_text(json.dumps({'complete':not bool(cli.limit),'processed':offset,'f1':scores,'paired':paired,'note':'DEV fixed-checkpoint source swap; conditional effects and interactions, not training ablation. Same donors as earlier audit. Samples retain image IDs and changed candidate spans.'},indent=2))
    print(scores)

if __name__=='__main__':main()
