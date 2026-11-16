
"""Prespecified overnight exploration; frozen model, DEV only, serial per GPU."""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from run_attempt import ROOT, command

DEST = ROOT / 'night_explore'
# Leave a buffer before the user's 09:00 review. Running jobs are not killed.
CUTOFF = datetime.fromisoformat('2026-11-03T07:15:00+08:00').timestamp()

def jobs():
    result = []
    def add(family, variant, seed=42, overrides=None):
        result.append(dict(name=f'{family}_{variant}_s{seed}', family=family,
                           variant=variant, seed=seed, overrides=overrides or {}))
    for seed in [43, 44]:
        for variant in ['full', 'no_ib', 'no_alignment']:
            add('base', variant, seed)
    add('base', 'no_polar', overrides={'--beta_polar': '0'})
    add('base', 'full_mass', overrides={'--repair_transport': 'full'})
    # Temperature is inactive for none/uniform: reuse those existing controls.
    for variant in ['full', 'no_ib', 'vib', 'full_mass']:
        add('sharp', variant, overrides={'--entropy_eps': '0.03'})
    # no_ib has zero local KL and polarization, so reuse its base comparison.
    for variant in ['full', 'vib', 'no_alignment']:
        add('low_kl', variant, overrides={'--ib_kl_weight': '0.0001'})
    for variant in ['full', 'no_alignment', 'cosine_control']:
        add('strong_polar', variant, overrides={'--beta_polar': '0.05'})
    return result

def build(job, gpu):
    variant = job['variant']
    cmd = command('full' if variant in ['no_polar', 'full_mass'] else variant,
                  gpu, DEST/job['name'])
    variant_flags = {'--repair_transport': 'full'} if variant == 'full_mass' else {}
    for key, value in {'--seed': str(job['seed']), **variant_flags, **job['overrides']}.items():
        if key in cmd:
            for i, arg in enumerate(cmd[:-1]):
                if arg == key:
                    cmd[i+1] = value
        else:
            cmd.extend([key, value])
    return cmd

def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)

def summarize():
    rows = []
    for root in [ROOT/'runs', DEST]:
        for out in sorted(root.iterdir()):
            if not out.is_dir() or not (out/'status.json').exists():
                continue
            status = json.loads((out/'status.json').read_text())
            metrics = out/'eval_results.txt'
            scores = []
            if metrics.exists():
                for block in metrics.read_text().split('***** Eval results dev ')[1:]:
                    match = re.search(r'^span_f1 = ([0-9.eE+-]+)', block, re.M)
                    if match:
                        scores.append(float(match[1]))
            probe = out/'mechanism_probe.jsonl'
            diagnostics = []
            if probe.exists():
                for line in probe.read_text().splitlines():
                    try:
                        diagnostics.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # A worker may still be writing the last line.
            latest = diagnostics[-1] if diagnostics else {}
            row = dict(run=out.name, directory=str(out), state=status['state'],
                       epochs_evaluated=len(scores), best_dev_f1=max(scores) if scores else None,
                       best_epoch=scores.index(max(scores))+1 if scores else None,
                       last_dev_f1=scores[-1] if scores else None,
                       last_step=latest.get('step'), last_loss=latest.get('loss'),
                       text_gate_mean=latest.get('text_gate_mean'),
                       image_gate_mean=latest.get('image_gate_mean'),
                       max_residual=max((d.get('max_capacity_mass_residual', 0)
                                         for d in diagnostics), default=None))
            rows.append(row)
    write_json(DEST/'summary.json', rows)
    if rows:
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temp = DEST/'summary.csv.tmp'
        temp.write_text(stream.getvalue())
        temp.replace(DEST/'summary.csv')

def predecessor_running(gpu):
    entries = json.loads((ROOT/'runs/launch.json').read_text())
    item = next(entry for entry in entries if entry['gpu'] == gpu)
    try:
        cmdline = Path(f"/proc/{item['pid']}/cmdline").read_bytes()
    except FileNotFoundError:
        return False
    return b'run_attempt.py' in cmdline and b'--worker' in cmdline

def check_sources():
    hashes = json.loads((ROOT/'runs/source_hashes.json').read_text())
    for rel, digest in hashes.items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Source changed after initial experiment launch: {rel}')

def scheduler():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        raise RuntimeError('Three GPUs required; no CPU fallback')
    check_sources()
    pending = jobs()
    active = {}
    finished = []
    while pending or active or any(predecessor_running(gpu) for gpu in range(3)):
        for gpu, (proc, job, started, handle) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            handle.close()
            write_json(DEST/job['name']/'status.json', dict(
                state='complete' if code == 0 else 'failed', returncode=code,
                start=started, end=time.time(), gpu=gpu, **job))
            finished.append(dict(name=job['name'], returncode=code))
            del active[gpu]
        for gpu in range(3):
            if not pending or gpu in active or predecessor_running(gpu):
                continue
            # Reserve space for active jobs' checkpoints and 6 GiB free space.
            other_count = len(active) + sum(predecessor_running(g) for g in range(3))
            required = (6 + 2 * (other_count + 1)) * 1024**3
            if time.time() >= CUTOFF:
                for job in pending:
                    job['deferred_reason'] = '07:15 cutoff reached; do not launch new jobs'
                break
            if shutil.disk_usage(ROOT).free < required:
                for job in pending:
                    job['deferred_reason'] = 'disk reserve; no existing results deleted'
                break
            check_sources()
            job = pending.pop(0)
            job.pop('deferred_reason', None)
            out = DEST/job['name']
            out.mkdir(exist_ok=False)
            cmd = build(job, gpu)
            write_json(out/'command.json', cmd)
            started = time.time()
            write_json(out/'status.json', dict(state='running', start=started, gpu=gpu, **job))
            handle = (out/'train.log').open('w')
            env = {**os.environ, 'PYTHONHASHSEED': str(job['seed']),
                   'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
            proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                    stdout=handle, stderr=subprocess.STDOUT, env=env)
            active[gpu] = (proc, job, started, handle)
        summarize()
        waiting_original = any(predecessor_running(g) for g in range(3))
        stopped = bool(pending and all('deferred_reason' in j for j in pending)
                       and not active and not waiting_original)
        write_json(DEST/'queue_status.json', dict(
            updated=time.time(), active={g: {'name': v[1]['name'], 'pid': v[0].pid}
                                        for g, v in active.items()},
            pending=pending, finished=finished,
            waiting_for_initial_runs=waiting_original,
            state='deferred' if stopped else 'running' if pending or active or waiting_original else 'complete'))
        if stopped:
            break
        if pending or active or waiting_original:
            time.sleep(30)
    summarize()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    if args.summary:
        summarize()
    elif args.worker:
        scheduler()
    elif args.launch:
        check_sources()
        DEST.mkdir(exist_ok=False)
        write_json(DEST/'protocol.json', dict(
            jobs=jobs(), epochs=8, test_evaluation=False, checkpoint_selection='DEV',
            cutoff='2026-11-03 07:15 Asia/Shanghai', seeds=[42,43,44],
            priority='core paired multi-seed, mechanism controls, exploratory candidates',
            score_based_cancellation=False,
            caveat='8-epoch exploratory runs, no guarantee of convergence or positive ablations'))
        shutil.copy2(ROOT/'runs/source_hashes.json', DEST/'source_hashes.json')
        shutil.copy2(Path(__file__), DEST/'scheduler_snapshot.py')
        with (DEST/'scheduler.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        write_json(DEST/'launch.json', dict(pid=proc.pid, started=time.time()))
        print(json.dumps({'pid': proc.pid, 'additional_jobs': len(jobs()), 'output': str(DEST)}))
    else:
        print(json.dumps([dict(**job, command=build(job, 0)) for job in jobs()], indent=2))

if __name__ == '__main__':
    main()
