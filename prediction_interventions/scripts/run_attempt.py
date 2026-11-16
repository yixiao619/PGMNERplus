"""Frozen exploratory DEV-only queue; one worker per GPU, no score-based stops."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    'full': [],
    'no_ib': ['--repair_ib', 'off'],
    'no_alignment': ['--repair_transport', 'none'],
    'cosine_control': ['--repair_cost', 'cosine'],
    'uniform_partial': ['--repair_transport', 'uniform_partial'],
    'vib': ['--repair_ib', 'vib'],
}
QUEUES = [['full', 'cosine_control'], ['no_ib', 'vib'],
          ['no_alignment', 'uniform_partial']]

def command(variant, gpu, out, smoke=False):
    cmd = json.loads((ROOT/'scripts/base_command.json').read_text())
    cmd[0] = sys.executable
    updates = {'--epoch': '8', '--repair_stop_epochs': '0', '--seed': '42',
               '--gpu_id': str(gpu), '--output_dir': str(out),
               '--lr': '1e-5', '--bert_lr': '1e-5', '--beta_polar': '0.01'}
    for key, value in updates.items():
        cmd[cmd.index(key)+1] = value
    cmd += ['--repair_safe_fusion', '--repair_cost', 'gate_reliability']
    cmd += VARIANTS[variant]
    if smoke:
        cmd += ['--repair_smoke_steps', '3']
    return cmd

def run_one(variant, gpu, smoke):
    import torch
    if not torch.cuda.is_available() or gpu >= torch.cuda.device_count():
        raise RuntimeError('Requested GPU unavailable; refusing silent CPU training')
    out = ROOT / ('smoke_runs' if smoke else 'runs') / f'{variant}_s42'
    out.mkdir(parents=True, exist_ok=False)
    cmd = command(variant, gpu, out, smoke)
    (out/'command.json').write_text(json.dumps(cmd, indent=2))
    status = {'state': 'running', 'start': time.time(), 'gpu': gpu}
    (out/'status.json').write_text(json.dumps(status, indent=2))
    env = {**os.environ, 'PYTHONHASHSEED': '42', 'PYTHONUNBUFFERED': '1',
           'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    with (out/'train.log').open('w') as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
    status.update(state='complete' if result.returncode == 0 else 'failed',
                  returncode=result.returncode, end=time.time())
    (out/'status.json').write_text(json.dumps(status, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, choices=[0, 1, 2])
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker is not None:
        for variant in QUEUES[args.worker]:
            run_one(variant, args.worker, args.smoke)
        return
    if not args.launch:
        for gpu, queue in enumerate(QUEUES):
            for variant in queue:
                print(json.dumps(command(variant, gpu, ROOT/'runs'/f'{variant}_s42', args.smoke)))
        return
    dest = ROOT/('smoke_runs' if args.smoke else 'runs')
    dest.mkdir(exist_ok=False)
    files = list(ROOT.glob('*.py'))
    for folder in ['models', 'utils', 'scripts']:
        files += list((ROOT/folder).rglob('*.py'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(files)}
    (dest/'source_hashes.json').write_text(json.dumps(hashes, indent=2))
    (dest/'protocol.json').write_text(json.dumps({
        'seed': 42, 'epochs': 8, 'selection': 'DEV', 'test_evaluation': False,
        'purpose': 'exploratory gate-cost repair; not confirmatory evidence',
        'queues': QUEUES, 'score_based_stopping': False,
        'common_changes': ['lr=bert_lr=1e-5', 'safe_fusion', 'gated_posterior',
                           'ot_alpha=0', '8-epoch scheduler'],
    }, indent=2))
    processes = []
    for gpu in range(3):
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', str(gpu)]
        if args.smoke:
            cmd += ['--smoke']
        with (dest/f'worker_gpu{gpu}.log').open('w') as log:
            child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        processes.append({'gpu': gpu, 'pid': child.pid, 'variants': QUEUES[gpu]})
    (dest/'launch.json').write_text(json.dumps(processes, indent=2))
    print(json.dumps(processes, indent=2))

if __name__ == '__main__':
    main()
