"""Three seed-42 training controls followed by the three mechanism experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path('/home/jyx/PGMNER_positive_seed42_20261103')
TRAIN = ['balanced_full', 'balanced_no_polarization', 'balanced_no_alignment']
OLD = ['sharp_full_s42', 'sharp_no_ib_s42', 'sharp_vib_s42']

def dump(path, value):
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)

def command(gpu, smoke):
    out = ROOT/('smoke_runs' if smoke else 'runs')/TRAIN[gpu]
    cmd = json.loads((ARCHIVE/'selected/sharp_full_s42/command.json').read_text())
    cmd[0] = sys.executable
    for key, value in {'--gpu_id': str(gpu), '--output_dir': str(out)}.items():
        cmd[cmd.index(key)+1] = value
    cmd += ['--repair_gate_balance', 'bernoulli', '--repair_gate_target', '0.6',
            '--repair_gate_balance_weight', '2', '--repair_polarization_weight',
            '0' if gpu == 1 else '1']
    if gpu == 2:
        cmd += ['--repair_transport', 'none']
    if smoke:
        cmd += ['--repair_smoke_steps', '3']
    return out, cmd

def summarize():
    result = {'training': [], 'audits': []}
    for root in [ROOT/'runs', ROOT/'audits']:
        if not root.exists(): continue
        for out in sorted(root.iterdir()):
            if not out.is_dir(): continue
            row = {'name': out.name}
            for name in ['status.json', 'results.json']:
                if (out/name).exists(): row[name[:-5]] = json.loads((out/name).read_text())
            if (out/'eval_results.txt').exists():
                scores = list(map(float, re.findall(r'^span_f1 = ([0-9.]+)',
                              (out/'eval_results.txt').read_text(), re.M)))
                row['dev_curve'] = scores
                row['best_dev_f1'] = max(scores) if scores else None
            result['training' if root.name == 'runs' else 'audits'].append(row)
    dump(ROOT/'study_summary.json', result)

def run_process(cmd, logpath, env):
    with logpath.open('w') as log:
        return subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                              env=env).returncode

def worker(gpu, smoke):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        raise RuntimeError('Three visible GPUs required')
    base = ROOT/('smoke_runs' if smoke else 'runs')
    hashes = json.loads((base/'source_hashes.json').read_text())
    for rel, digest in hashes.items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
    out, cmd = command(gpu, smoke)
    out.mkdir(exist_ok=False)
    dump(out/'command.json', cmd)
    env = {**os.environ, 'PYTHONHASHSEED': '42', 'PYTHONUNBUFFERED': '1',
           'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1'}
    status = dict(state='running', start=time.time(), gpu=gpu)
    dump(out/'status.json', status)
    code = run_process(cmd, out/'train.log', env)
    status.update(state='complete' if code == 0 else 'failed', end=time.time(), returncode=code)
    dump(out/'status.json', status)
    if not smoke: summarize()
    if code or smoke: return code
    for checkpoint, name in [(out, TRAIN[gpu]), (ARCHIVE/'selected'/OLD[gpu], 'legacy_'+OLD[gpu])]:
        audit = ROOT/'audits'/name
        cmd = [sys.executable, str(ROOT/'scripts/mechanism_audit.py'), '--run', str(checkpoint),
               '--output', str(audit), '--gpu', str(gpu)]
        audit_status = dict(state='running', start=time.time(), gpu=gpu, command=cmd)
        dump(ROOT/'audits'/f'{name}_status.json', audit_status)
        code = run_process(cmd, ROOT/'audits'/f'{name}.log', env)
        audit_status.update(state='complete' if code == 0 else 'failed', end=time.time(), returncode=code)
        dump(ROOT/'audits'/f'{name}_status.json', audit_status)
        if audit.exists(): dump(audit/'status.json', audit_status)
        if code == 0:
            plot_code = run_process([sys.executable, str(ROOT/'scripts/plot_audit.py'), str(audit)],
                                    ROOT/'audits'/f'{name}_plot.log', env)
            dump(audit/'plot_status.json', {'returncode': plot_code})
        summarize()
        # Retain failed audit and continue independent comparison checkpoints.
    return 0

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--launch', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--worker', type=int, choices=[0,1,2])
    p.add_argument('--summary', action='store_true')
    args = p.parse_args()
    if args.summary:
        summarize(); return
    if args.worker is not None:
        raise SystemExit(worker(args.worker, args.smoke))
    if not args.launch:
        print(json.dumps([command(gpu,args.smoke)[1] for gpu in range(3)],indent=2)); return
    if not args.smoke and shutil.disk_usage(ROOT).free < 8*1024**3:
        raise RuntimeError('Need 8 GiB free for three checkpoints plus reserve; no automatic deletion')
    base = ROOT/('smoke_runs' if args.smoke else 'runs')
    base.mkdir(exist_ok=False)
    (ROOT/'audits').mkdir(exist_ok=True)
    files = list(ROOT.glob('*.py'))
    for folder in ['models', 'utils', 'scripts']:
        files += list((ROOT/folder).rglob('*.py'))
    dump(base/'source_hashes.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(files)})
    dump(base/'protocol.json', dict(seed=42, epochs=8, temperature=.03, test_evaluation=False,
        training=TRAIN, reference_checkpoints=OLD, gate_target=.6, balance_weight=2,
        interpretation='Single-seed exploratory mechanism study; no guaranteed positive direction',
        E1='Mean/distribution-preserving gate interventions',
        E2='Marginal-preserving assignment removal and wrong-image control',
        E3='2x2 gate/assignment intervention at 0/25/50/75% donor-patch corruption',
        selection='DEV best checkpoint; audit conditions not used for checkpoint selection'))
    children = []
    for gpu in range(3):
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', str(gpu)]
        if args.smoke: cmd.append('--smoke')
        with (base/f'gpu{gpu}.log').open('w') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        children.append(dict(gpu=gpu, pid=proc.pid, training=TRAIN[gpu]))
    dump(base/'launch.json', children)
    print(json.dumps(children, indent=2))

if __name__ == '__main__':
    main()
