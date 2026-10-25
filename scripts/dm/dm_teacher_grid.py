"""One teacher parameter at a time around the defaults, on the first 10 val matches.

A value replaces the default only if the paired frag difference has its lower 95% bound above 0 and
K/D does not drop.

    python scripts/dm/dm_teacher_grid.py  ->  reports/dm/v1/d1_teacher/d1_grid.md
"""
import argparse
import sys
from dataclasses import replace

from doomagent.dm.config import DM, VAL_SEEDS
from doomagent.dm.evaluate import evaluate_plan, summarize
from doomagent.dm.teacher import DMTeacher
from doomagent.evaluate import paired_diff_ci
from doomagent.variants import BASE

GRID = {'safe_dist': (128.0, 224.0), 'chase_s': (1.0, 3.0), 'low_health': (30.0, 70.0),
        'low_ammo': (3.0, 8.0), 'aim_margin': (0.0, 0.35)}

if __name__ == '__main__':  # worker processes re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default='reports/dm/v1/d1_teacher/d1_grid.md')
    a = ap.parse_args()
    plan = [(BASE, VAL_SEEDS[:10])]
    base = evaluate_plan(plan, DMTeacher(DM), DM, workers=a.workers)
    sb = summarize(base)
    md = ['# D1: teacher parameter grid (10 val matches each)', '', f'Command: `python {" ".join(sys.argv)}`',
          '', f'Defaults: frags {sb["frags"]["mean"]:.2f}, K/D total {sb["kd_total"]:.2f}', '',
          '| parameter | value | frags [95% CI] | vs default [95% CI] | K/D total | rule |',
          '| --- | --- | --- | --- | --- | --- |']
    chosen = {}
    for name, values in GRID.items():
        best = None
        for v in values:
            cfg = replace(DM, **{name: v})
            res = evaluate_plan(plan, DMTeacher(cfg), cfg, workers=a.workers)
            s = summarize(res)
            d = paired_diff_ci([r['frags'] for r in res], [r['frags'] for r in base])
            ok = d[1] > 0 and s['kd_total'] >= sb['kd_total']
            if ok and (best is None or d[0] > best[1]):
                best = (v, d[0])
            f = s['frags']
            md.append(f'| {name} | {v} | {f["mean"]:.2f} [{f["lo"]:.2f}, {f["hi"]:.2f}] | '
                      f'{d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}] | {s["kd_total"]:.2f} | '
                      f'{"better" if ok else "keep default"} |')
            print(md[-1], flush=True)
        if best is not None:
            chosen[name] = best[0]
    md += ['', 'Chosen: ' + (', '.join(f'{k}={v}' for k, v in chosen.items()) or 'none (defaults stay)')]
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(md) + '\n')
    print(md[-1])
