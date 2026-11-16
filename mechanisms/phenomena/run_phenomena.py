"""Small mechanism demonstrations; synthetic evidence is explicitly labeled."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime

import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models.repaired_alignment import LocalPolarizedIB, FixedMassTransport

torch.set_num_threads(1)
torch.manual_seed(42)

def save(fig, name):
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(DEST/f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)

def saturation():
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    x = torch.zeros(1, 16, 4)
    mask = torch.ones(1, 16)
    for ax, init in zip(axes, [-12., 12.]):
        for mode in ['legacy', 'bernoulli']:
            ib = LocalPolarizedIB(4).eval()
            ib.balance_mode = mode
            with torch.no_grad():
                ib.gate.weight.zero_(); ib.gate.bias.fill_(init)
            opt = torch.optim.SGD(ib.gate.parameters(), lr=.25)
            trace = []
            for step in range(241):
                opt.zero_grad()
                loss = ib(x, mask)[2]
                loss.backward()
                trace.append(dict(step=step, mean=float(ib.last_gate.mean()),
                                  bias=float(ib.gate.bias.detach()), gradient=float(ib.gate.bias.grad),
                                  objective=float(loss.detach())))
                if step < 240: opt.step()
            records.append(dict(mode=mode, initial_logit=init, trace=trace))
            ax.plot([v['step'] for v in trace], [v['mean'] for v in trace], label=mode)
        ax.axhline(.6, linestyle=':', color='gray', label='soft-budget target')
        ax.set(title=f'Synthetic uniform gates: initial logit {init:g}',
               xlabel='Regularizer-only SGD steps', ylabel='Mean gate', ylim=(-.03, 1.03))
        ax.legend(fontsize=8)
    save(fig, 'A_saturated_recovery')
    return records

def training_snapshot():
    runs = ['balanced_full', 'balanced_no_polarization']
    probes = {}
    hashes = {}
    for name in runs:
        data = (ROOT/'runs'/name/'mechanism_probe.jsonl').read_bytes()
        # A trailing partial line can occur while the writer is active.
        parsed = []
        for line in data.splitlines():
            try: parsed.append(json.loads(line))
            except json.JSONDecodeError: pass
        probes[name] = {v['step']: v for v in parsed}
        (DEST/f'{name}_probe_snapshot.json').write_text(json.dumps(parsed, indent=2))
        hashes[name] = hashlib.sha256(data).hexdigest()
    common = sorted(set(probes[runs[0]]) & set(probes[runs[1]]))
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    for row, modality in enumerate(['text', 'image']):
        for col, statistic in enumerate(['mean', 'std']):
            ax = axes[row, col]
            for name in runs:
                values = [probes[name][step][f'{modality}_gate_{statistic}'] for step in common]
                ax.plot(common, values, label=name.replace('balanced_', ''), alpha=.8)
            if statistic == 'mean': ax.axhline(.6, color='gray', linestyle=':')
            ax.set(xlabel='Matched optimizer step', ylabel=f'{modality} gate {statistic}', ylim=(0, 1))
            ax.legend(fontsize=8)
    fig.suptitle('Real TRAIN probes, seed 42: equal soft budget; ongoing snapshot')
    save(fig, 'B_real_gate_trajectories')
    # Last five matched probes reduce the dependence on one minibatch.
    tail = common[-5:]
    summary = {name: {f'{mod}_{stat}': float(np.mean([
        probes[name][step][f'{mod}_gate_{stat}'] for step in tail]))
        for mod in ['text', 'image'] for stat in ['mean', 'std', 'low_fraction', 'high_fraction']}
        for name in runs}
    return dict(timestamp=datetime.now().astimezone().isoformat(), matched_steps=common,
                last_five_steps=tail, summary=summary, snapshot_source_sha256=hashes,
                caveat='Within-minibatch SD is not a full-dataset or per-sample distribution estimate')

def solve(cost, eps, full=False):
    # Numerical budget can increase only to meet a prespecified residual, never
    # according to a desired substantive result.
    for iterations in [100, 500, 2000, 6000]:
        solver = FixedMassTransport(eps=eps, n_iters=iterations, mass=.6).double()
        with torch.no_grad():
            plan, _ = solver(cost[None], torch.ones(1, cost.shape[0]),
                             torch.ones(1, cost.shape[1]), full=full)
        residual = float(solver.last_residual.max())
        if residual < 1e-7:
            return plan[0], residual, iterations
    raise RuntimeError(f'Toy solver did not converge: {residual}')

def distractors():
    records = []
    good = 6
    for bad in [0, 2, 4, 6, 8]:
        # Genuine prototypes are orthogonal and matched exactly. Distractors
        # are orthogonal to all text features by construction, not discovered.
        basis = torch.eye(good+bad, dtype=torch.float64)
        t, v = basis[:good], basis
        cost = 1-t@v.T
        for kind in ['partial', 'full', 'uniform_partial']:
            if kind == 'uniform_partial':
                p = torch.full((good,good+bad), .6/(good*(good+bad)), dtype=torch.float64)
                residual, iters = 0., 0
            else:
                p, residual, iters = solve(cost, .08, full=kind=='full')
            mass = float(p.sum())
            noise_mass = float(p[:, good:].sum())
            records.append(dict(distractors=bad, mode=kind, total_mass=mass,
                distractor_mass=noise_mass, distractor_fraction=noise_mass/mass,
                correct_diagonal_mass=float(p[:, :good].diag().sum()),
                forced_distractor_lower_bound=max(0.,mass-good/(good+bad)),
                max_residual=residual, iterations=iters))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for kind in ['partial', 'full', 'uniform_partial']:
        values = [v for v in records if v['mode']==kind]
        axes[0].plot([v['distractors'] for v in values], [v['distractor_fraction'] for v in values],
                     marker='o', label=kind)
        axes[1].plot([v['distractors'] for v in values], [v['correct_diagonal_mass']/v['total_mass'] for v in values],
                     marker='o', label=kind)
    pp = [v for v in records if v['mode']=='partial']
    axes[0].plot([v['distractors'] for v in pp],
                 [v['forced_distractor_lower_bound']/v['total_mass'] for v in pp],
                 linestyle=':', color='black', label='partial capacity lower bound')
    for ax in axes:
        ax.set(xlabel='Number of orthogonal distractors', ylim=(-.02,1.02))
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Fraction of transported mass on distractors')
    axes[1].set_ylabel('Fraction of mass on exact prototype matches')
    fig.suptitle('Synthetic transport: 6 known matches; s=0.6, epsilon=0.08')
    save(fig, 'C_partial_transport_rejection')
    return records

def gate_to_plan():
    t = torch.eye(4, dtype=torch.float64)[:3]
    v = torch.eye(4, dtype=torch.float64)
    records, heatmaps = [], {}
    for mode in ['cosine', 'gate_reliability']:
        for value in [.1, .3, .5, .7, .9]:
            gt = torch.full((3,1), .8, dtype=torch.float64)
            gv = torch.full((4,1), .8, dtype=torch.float64)
            gv[0] = value
            similarity = F.normalize(gt*t, dim=-1) @ F.normalize(gv*v, dim=-1).T
            cost = 1-similarity
            if mode == 'gate_reliability':
                reliability = gt@gv.T
                cost = reliability*cost+(1-reliability)*2
            plan, residual, iters = solve(cost, .03)
            records.append(dict(mode=mode, visual_gate=value, mass_on_patch0=float(plan[:,0].sum()),
                                plan=plan.tolist(), max_residual=residual, iterations=iters))
            if value in [.1,.9]: heatmaps[(mode,value)] = plan.numpy()
    fig, axes = plt.subplots(2,3,figsize=(12,6),gridspec_kw={'width_ratios':[1,1,1.3]})
    for row, mode in enumerate(['cosine','gate_reliability']):
        for col,value in enumerate([.1,.9]):
            axes[row,col].imshow(heatmaps[(mode,value)],vmin=0,vmax=.25,cmap='viridis',aspect='auto')
            axes[row,col].set(title=f'{mode}: image gate[0]={value}',xlabel='Visual token',ylabel='Text token')
        rr=[r for r in records if r['mode']==mode]
        axes[row,2].plot([r['visual_gate'] for r in rr],[r['mass_on_patch0'] for r in rr],marker='o')
        axes[row,2].set(xlabel='Intervened gate on patch 0',ylabel='Mass allocated to patch 0',ylim=(0,.26))
    fig.suptitle('Synthetic gate intervention; identical feature directions, fixed transport budget')
    save(fig,'D_gate_to_matching')
    return records

if __name__ == '__main__':
    target=DEST/'results.json'
    if target.exists(): raise SystemExit('Refuse to overwrite a recorded snapshot')
    results={}
    for name, fn in [('A',saturation),('B',training_snapshot),('C',distractors),('D',gate_to_plan)]:
        results[name]=fn()
        (DEST/f'{name}_results.json').write_text(json.dumps(results[name],indent=2))
        print(name,'complete',flush=True)
    target.write_text(json.dumps(results,indent=2))
    (DEST/'code_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve(),ROOT/'models/repaired_alignment.py']},indent=2))
