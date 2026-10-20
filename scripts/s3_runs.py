"""S3 queue: bc0 and its ablations (plan, Task 8 Step 5), N runs at a time, on any OS.

A run whose log already reached the last epoch is skipped, so the queue can be restarted.
"""
import argparse
import json
import os
import subprocess
import sys
import time

RUNS = [('bc0', []),
        ('bc0_noaux', ['--lam', '0', '--lam-seg', '0']),
        ('bc0_noseg', ['--lam-seg', '0']),
        ('bc0_nohflip', ['--no-hflip']),
        ('bc0_w1.5', ['--shoot-w', '1.5']),
        ('bc0_w2.0', ['--shoot-w', '2.0']),
        ('bc0_aug', ['--aug'])]


def finished(name, epochs):
    try:
        with open(f'runs/{name}.pt.log.json') as fh:
            return json.load(fh)['log'][-1]['epoch'] >= epochs
    except (OSError, ValueError, KeyError, IndexError):
        return False


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--parallel', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--workers', type=int, default=0, help='DataLoader workers per run')
    ap.add_argument('--only', nargs='*', help='run names to keep, e.g. bc0 bc0_aug')
    ap.add_argument('dirs', nargs='*', default=['data/train'])
    a = ap.parse_args()
    os.makedirs('runs', exist_ok=True)
    todo = [(n, f) for n, f in RUNS if (not a.only or n in a.only) and not finished(n, a.epochs)]
    running = []
    while todo or running:
        while todo and len(running) < a.parallel:
            name, flags = todo.pop(0)
            cmd = [sys.executable, '-u', '-m', 'doomagent.train_bc', *a.dirs, '--out',
                   f'runs/{name}.pt', '--epochs', str(a.epochs), '--workers', str(a.workers), *flags]
            log = open(f'runs/{name}.out', 'w')
            running.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
            print(time.strftime('%H:%M:%S'), 'start', name, flush=True)
        for item in list(running):
            name, proc, log = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                status = 'done' if proc.returncode == 0 else f'FAILED (code {proc.returncode})'
                print(time.strftime('%H:%M:%S'), status, name, flush=True)
        time.sleep(2)
