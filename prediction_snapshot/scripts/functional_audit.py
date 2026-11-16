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
    rng = np.random.default_rng(42)
    # A fixed label-independent donor for every post, with a different image ID.
    donor = np.empty(n, dtype=int)
    image_ids = np.array([str(e.imgid) for e in examples])
    for i in range(n):
        eligible = np.flatnonzero(image_ids != image_ids[i])
        if not len(eligible): raise ValueError('No distinct donor image')
        donor[i] = rng.choice(eligible)
    (dest/'donors.json').write_text(json.dumps(donor.tolist()))
    rows = conditions()
    if aligner.ib=='polarized':
        rows += [dict(name=mode,noise=0.,gate=mode,plan='native',wrong_image=False) for mode in ['keep_high','keep_low','keep_random']]
    (dest/'protocol.json').write_text(json.dumps(dict(
        seed=42, split='dev', conditions=rows, limit=cli.limit, checkpoint=str(run),
        primary=['E1_gate_assignment', 'E2_transport_assignment', 'E3_interaction_at_50pct_noise'],
        uncertainty='Paired post bootstrap, 1000 resamples; descriptive, no training-seed uncertainty',
        corruption='Replace 25/50/75% of non-CLS encoded patches by different-image patches; nested masks',
        caveat='Inference perturbations test dependence, not a retrained architectural ablation'), indent=2))
    current = dict(rows[0], batch_index=0)
    def gate_hook(which):
        def hook(module, inputs, result):
            name = current['gate']
            selected = ((which=='image' and name.startswith('keep_')) or name in ['mean', 'shuffle_both'] or
                        name == ('shuffle_text' if which == 'text' else 'shuffle_image'))
            if not selected:
                return result
            x, mask = inputs[:2]
            g = change_gate(module.current_gate, mask,
                            name if name.startswith('keep_') else ('mean' if name == 'mean' else 'shuffle'),
                            42 + current['batch_index']*100 + (0 if which == 'text' else 1))
            module.current_gate = g  # Cost AND values must see the same intervention.
            module.last_gate = g.detach()
            return g*module.mu(x)*mask.unsqueeze(-1), result[1], result[2]
        return hook
    aligner.ib_t.register_forward_hook(gate_hook('text'))
    aligner.ib_v.register_forward_hook(gate_hook('image'))
    max_marginal_error = [0.]
    rejection_records=[]
    def plan_hook(module, inputs, result):
        if current['gate']=='native' and current['plan']=='native':
            plan=result[0];cost,mt,mv=inputs[:3]
            for i in range(len(plan)):
                valid=mt[i].bool();vv=mv[i].bool();capacity=1/int(valid.sum());fraction=plan[i].sum(-1)[valid]/capacity
                best=cost[i][valid][:,vv].min(-1).values;order=best.argsort();k=max(1,len(order)//4)
                rejection_records.append({'condition':current['name'],'batch':current['batch_index'],'sample':i,'mass':float(plan[i].sum()),'mean_unmatched_fraction':float((1-fraction).mean()),'low_cost_unmatched':float((1-fraction)[order[:k]].mean()),'high_cost_unmatched':float((1-fraction)[order[-k:]].mean()),'max_row_excess':float((fraction-1).clamp_min(0).max())})
        if current['plan'] == 'native': return result
        p, loss = result
        q = decorrelate_plan(p)
        error = max(float((p.sum(1)-q.sum(1)).abs().max()),
                    float((p.sum(2)-q.sum(2)).abs().max()))
        max_marginal_error[0] = max(max_marginal_error[0], error)
        assert error < 1e-5
        return q, loss
    aligner.solver.register_forward_hook(plan_hook)
    counts = {row['name']: [] for row in rows}
    sensitivity = {row['name']: {'probability_l1_sum':0., 'changed_argmax':0, 'valid_spans':0} for row in rows}
    gate_stats = []
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, shuffle=False)
    offset = 0
    with torch.inference_mode():
        for batch_index, cpu in enumerate(loader):
            batch = {k: v.to(pg.args.device) for k, v in cpu.items()}
            bs = len(batch['input_ids'])
            text, _, attention = model.span_encoding(**batch)
            _, visual = model.image_model(batch['image'])
            donor_cpu = collate_fn([dataset[int(donor[i])] for i in range(offset, offset+bs)])
            _, donor_visual = model.image_model(donor_cpu['image'].to(pg.args.device))
            patch_count = visual.shape[1]-1  # Keep CLS unchanged under patch corruption.
            ranks = []
            for i in range(offset, offset+bs):
                ranks.append(np.argsort(np.random.default_rng(42+i).permutation(patch_count)))
            rank = torch.tensor(np.stack(ranks), device=visual.device)
            native = None
            for row in rows:
                current.update(row, batch_index=batch_index)
                corruption = torch.cat([torch.zeros(bs, 1, dtype=torch.bool, device=visual.device),
                                        rank < round(row['noise']*patch_count)], 1)
                image = donor_visual if row['wrong_image'] else torch.where(
                    corruption.unsqueeze(-1), donor_visual, visual)
                out = aligner(text, image, mask_t=batch['span_masks'].float(), mask_v=None,
                              attn_t=attention, span_token_idxes=batch.get('span_token_idxes'))
                t, _, at = out[:3]
                logits = model.span_classifier(model.ln_text(model.fusion_text(torch.cat([t, at], -1))+at))
                if getattr(model, '_misc_logit_bias', None) is not None and model.id_misc_focal >= 0:
                    logits[:, :, model.id_misc_focal] += float(model._misc_logit_bias)
                probs = model.softmax(logits)
                if row['name'] == 'n0_native_native':
                    native = probs
                    if batch_index == 0:
                        reference, _ = model({**batch, 'span_labels': None})
                        assert torch.allclose(probs, reference[0], atol=2e-6), 'Cached path mismatch'
                valid=batch['span_masks'].bool()
                record=sensitivity[row['name']]
                record['probability_l1_sum'] += float((probs-native).abs().sum(-1)[valid].sum())
                record['changed_argmax'] += int((probs.argmax(-1)[valid]!=native.argmax(-1)[valid]).sum())
                record['valid_spans'] += int(valid.sum())
                # Use the exact training evaluator on CPU, retaining per-post counts.
                pp = probs.cpu()
                for i in range(bs):
                    cc, _ = pg.span_f1_prune(cpu['span_word_idxes'][i:i+1], pp[i:i+1],
                                            cpu['span_labels'][i:i+1], cpu['span_masks'][i:i+1])
                    counts[row['name']].append(cc.numpy())
                if row['gate'] == 'native' and row['plan'] == 'native' and aligner.ib != 'off':
                    record = dict(condition=row['name'], batch=batch_index, n_posts=bs)
                    for modality, ib, mask in [('text', aligner.ib_t, batch['span_masks'].bool()),
                                                ('image', aligner.ib_v, torch.ones_like(corruption))]:
                        g = ib.last_gate.squeeze(-1)[mask]
                        record[modality] = dict(n=len(g), mean=float(g.mean()),
                            std=float(g.std(unbiased=False)), low=float((g<.1).float().mean()),
                            high=float((g>.9).float().mean()))
                    if corruption.any():
                        gv = aligner.ib_v.last_gate.squeeze(-1)
                        record['corrupt_gate_mean'] = float(gv[corruption].mean())
                        record['retained_patch_gate_mean'] = float(gv[:, 1:][~corruption[:, 1:]].mean())
                        mass = out[4].sum(1)
                        record['mass_per_corrupt_patch'] = float(mass[corruption].mean())
                        record['mass_per_retained_patch'] = float(mass[:, 1:][~corruption[:, 1:]].mean())
                    gate_stats.append(record)
            offset += bs
            (dest/'progress.json').write_text(json.dumps(dict(processed=offset, total=n)))
            print('audited', offset, '/', n, flush=True)
            if cli.limit and offset >= cli.limit: break
    arrays = {k: np.stack(v) for k, v in counts.items()}
    # no_gota still has cross-modal PPIB context; it is not text-only.
    if aligner.ib in ['off', 'vib']:
        assert np.array_equal(arrays['n0_native_native'], arrays['n0_shuffle_both_native']), 'Unit/absent gates must be shuffle-invariant'
    np.savez_compressed(dest/'per_post_counts.npz', **arrays)
    native_score = float(f1(arrays['n0_native_native']))
    if not cli.limit:
        expected = max(map(float, re.findall(r'^span_f1 = ([0-9.]+)',
                       (run/'eval_results.txt').read_text(), re.M)))
        assert abs(native_score-expected) < 1.1e-5, (native_score, expected)
    results = dict(complete=not bool(cli.limit), processed=offset,
                   f1={k: float(f1(v)) for k, v in arrays.items()},
                   sensitivity=sensitivity, paired=paired_estimates(arrays), max_marginal_error=max_marginal_error[0])
    (dest/'rejection_records.json').write_text(json.dumps(rejection_records,indent=2))
    (dest/'results.json').write_text(json.dumps(results, indent=2))
    (dest/'gate_statistics.json').write_text(json.dumps(gate_stats, indent=2))
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
