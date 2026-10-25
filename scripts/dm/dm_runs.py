"""E4 queue: behavior cloning of M0 and M1 on R0 (phase 1) and R0 + R1 (phase 2), N runs at a time.

A run whose log reached the last epoch and whose model file exists is skipped, so the queue can be restarted.

    python scripts/dm/dm_runs.py --phase 1 --parallel 1
    python scripts/dm/dm_runs.py --phase 2 --parallel 1 [--conv]
    python scripts/dm/dm_runs.py --phase 3 --parallel 1
    python scripts/dm/dm_runs.py --phase 4 --parallel 1
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = 'runs/dm'
R0, R01 = ['data/dm/r0'], ['data/dm/r0', 'data/dm/r1']
CONV = 'runs/ppo_ft_flip/best_model.zip'  # the first section's final model
PHASES = {1: [('dm2_m0_r0', 'm0', R0, []),
              ('dm2_m1_r0', 'm1', R0, []),
              ('dm2_m0_r0_conv', 'm0', R0, ['--conv-init', CONV])],
          2: [('dm2_m0_r01', 'm0', R01, []),
              ('dm2_m1_r01', 'm1', R01, [])],
          # after the inventory and the auxiliary targets (E4 decision): R0 of 300 matches + R1, conv init kept
          3: [('dm3_m1_aux', 'm1', R01, ['--conv-init', CONV, '--aux', '1.0']),
              ('dm3_m1', 'm1', R01, ['--conv-init', CONV]),
              ('dm3_m0_aux', 'm0', R01, ['--conv-init', CONV, '--aux', '1.0'])],
          # data re-recorded with the teacher's egomap and goal and the detour bursts: spatial heads against
          # the same run without them on the same data
          4: [('dm4_m1_sp', 'm1', R01, ['--conv-init', CONV, '--aux', '1.0', '--spatial', '1.0']),
              ('dm4_m1_aux', 'm1', R01, ['--conv-init', CONV, '--aux', '1.0']),
              ('dm4_m0_sp', 'm0', R01, ['--conv-init', CONV, '--aux', '1.0', '--spatial', '1.0'])]}


def finished(out, epochs):
    try:
        with open(out + '.log.json') as fh:
            return os.path.exists(out) and json.load(fh)['log'][-1]['epoch'] >= epochs
    except (OSError, ValueError, KeyError, IndexError):
        return False


def train_cmd(name, kind, dirs, flags, out, epochs, workers=2, eval_workers=4):
    return [sys.executable, '-u', '-m', 'doomagent.dm.train', kind, *dirs, '--out', out, '--epochs', str(epochs),
            '--workers', str(workers), '--eval-workers', str(eval_workers), *flags]


def run_queue(jobs, root=ROOT, epochs=12, parallel=1, make_cmd=train_cmd, poll_s=2.0):
    # jobs: [(name, kind, dirs, extra flags)]; each run's console goes to <root>/<name>.out
    os.makedirs(root, exist_ok=True)
    todo = [j for j in jobs if not finished(os.path.join(root, j[0] + '.pt'), epochs)]
    running, failed = [], []
    while todo or running:
        while todo and len(running) < parallel:
            name, kind, dirs, flags = todo.pop(0)
            out = os.path.join(root, name + '.pt')
            log = open(os.path.join(root, name + '.out'), 'w')
            proc = subprocess.Popen(make_cmd(name, kind, dirs, flags, out, epochs), stdout=log,
                                    stderr=subprocess.STDOUT)
            running.append((name, proc, log))
            print(time.strftime('%H:%M:%S'), 'start', name, flush=True)
        for item in list(running):
            name, proc, log = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                if proc.returncode:
                    failed.append(name)
                print(time.strftime('%H:%M:%S'), 'done' if not proc.returncode else
                      f'FAILED (code {proc.returncode})', name, flush=True)
        time.sleep(poll_s)
    return failed


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', type=int, choices=sorted(PHASES), required=True)
    ap.add_argument('--parallel', type=int, default=1, help='one at a time: two do not fit in 8 GB')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--workers', type=int, default=2, help='DataLoader workers per run')
    ap.add_argument('--eval-workers', type=int, default=4)
    ap.add_argument('--conv', action='store_true', help='phase 2 with the conv init (if the phase 1 rule kept it)')
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    jobs = [(n, k, d, f + (['--conv-init', CONV] if a.conv and a.phase == 2 else []))
            for n, k, d, f in PHASES[a.phase] if not a.only or n in a.only]

    def cmd(name, kind, dirs, flags, out, epochs):
        return train_cmd(name, kind, dirs, flags, out, epochs, a.workers, a.eval_workers)

    sys.exit(1 if run_queue(jobs, epochs=a.epochs, parallel=a.parallel, make_cmd=cmd) else 0)
