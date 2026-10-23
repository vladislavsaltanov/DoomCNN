"""One step of the teacher improvement cycle (plan v2, decision 9): a candidate change against the accepted
teacher on the val seeds, then on the fresh seeds; accepted only if the lower bound of the paired frag
difference on the fresh seeds is above 0. Appends a row to reports/dm/e1_teacher_cycle.md.

    python scripts/dm/dm_teacher_cycle.py --name hitscan_tol --base "" --cand "hitscan_tol=3.0" [--why "..."]

--base and --cand are space-separated name=value lists (the accepted teacher's settings and the candidate's);
diagnostic runs are cached in runs/dm/cycle/<key>_<seeds>.json.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

from doomagent.evaluate import bootstrap_ci

OUT = 'reports/dm/e1_teacher_cycle.md'
FRESH_N = 40  # all FRESH_SEEDS: 20 matches leave a ±1.8 frag interval, too wide for most changes
HEAD = ['# E1: teacher improvement cycle', '',
        'Each row: one change against the accepted teacher; 20 val matches, then 40 fresh matches (FRESH_SEEDS), '
        'paired by seed. A change is accepted only if the lower bound of the fresh difference is above 0 '
        '(plan v2, decision 9). Stop after two rejected changes in a row.', '',
        '| change | settings | why | val frags (accepted → candidate) | fresh frags (accepted → candidate) | '
        'fresh difference [95% CI] | best bot (cand.) | K/D (cand.) | first of 8 (cand.) | accepted |',
        '|---|---|---|---|---|---|---|---|---|---|']


def run(sets, seeds, workers):
    n = FRESH_N if seeds == 'fresh' else 20
    key = hashlib.md5(' '.join(sorted(sets)).encode()).hexdigest()[:10]
    path = f'runs/dm/cycle/{key}_{seeds}{n}'
    if not os.path.exists(path + '.json'):
        os.makedirs('runs/dm/cycle', exist_ok=True)
        cmd = [sys.executable, 'scripts/dm/dm_teacher_diag.py', '--seeds', seeds, '--matches', str(n), '--workers',
               str(workers), '--out', path] + (['--set'] + sets if sets else [])
        with open(path + '.out', 'w') as fh:
            subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT)
    return json.load(open(path + '.json'))


def frags(d):
    return {m['seed']: m['frags'] for m in d['per_match']}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # arrows in the row; the Windows console is cp1251
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--base', default='')
    ap.add_argument('--cand', required=True)
    ap.add_argument('--why', default='')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    base, cand = a.base.split(), a.cand.split()
    bv, cv = run(base, 'val', a.workers), run(cand, 'val', a.workers)
    bf, cf = run(base, 'fresh', a.workers), run(cand, 'fresh', a.workers)
    fb, fc = frags(bf), frags(cf)
    seeds = sorted(fb)
    m, lo, hi = bootstrap_ci(np.array([fc[s] - fb[s] for s in seeds]))
    ok = lo > 0
    s = cf['summary']
    row = (f'| {a.name} | `{" ".join(cand)}` | {a.why} | {bv["summary"]["frags"]:.2f} → {cv["summary"]["frags"]:.2f} | '
           f'{bf["summary"]["frags"]:.2f} → {s["frags"]:.2f} | {m:+.2f} [{lo:+.2f}, {hi:+.2f}] | {s["best_bot"]:.2f} | '
           f'{s["kd_total"]:.2f} | {100 * s["first_share"]:.0f}% | {"yes" if ok else "no"} |')
    lines = open(OUT, encoding='utf-8').read().rstrip('\n').split('\n') if os.path.exists(OUT) else HEAD
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines + [row]) + '\n')
    print(row)
    print('ACCEPTED' if ok else 'REJECTED')
