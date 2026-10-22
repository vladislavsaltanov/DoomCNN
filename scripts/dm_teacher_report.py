"""Teacher and random player on the val matches, Gate A.

    python scripts/dm_teacher_report.py [--set safe_dist=224] [--workers 4]
    ->  reports/d1_teacher.md, reports/d1_teacher.png, reports/d1_teacher.json
"""
import argparse
import json
import os
import sys
from dataclasses import fields, replace

import numpy as np

from doomagent.dm.config import DM, MODES, val_plan
from doomagent.dm.evaluate import RandomChooser, evaluate_plan, summarize
from doomagent.dm.reports import gate_a, match_table_md, teacher_figure
from doomagent.dm.teacher import DMTeacher


def with_overrides(cfg, items):
    # --set name=value pairs typed like the Cfg field
    types = {f.name: type(getattr(cfg, f.name)) for f in fields(cfg)}
    return replace(cfg, **{k: types[k](v) for k, v in (x.split('=', 1) for x in items or [])})


def mode_share(results):
    modes = np.concatenate([r['chooser']['modes'] for r in results])
    return {m: float((modes == i).mean()) for i, m in enumerate(MODES)}


if __name__ == '__main__':  # worker processes re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', action='append', help='Cfg override, name=value')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default='reports/d1_teacher')
    a = ap.parse_args()
    cfg = with_overrides(DM, a.set)
    plan = val_plan()
    teacher = evaluate_plan(plan, DMTeacher(cfg, keep_trace=True), cfg, workers=a.workers, trace=True)
    rand = evaluate_plan(plan, RandomChooser(), cfg, workers=a.workers)
    report = {'teacher': summarize(teacher), 'random': summarize(rand)}
    shares = mode_share(teacher)
    gate = gate_a(report)
    md = ['# D1: teacher on the val matches', '',
          f'Command: `python {" ".join(sys.argv)}`', '',
          f'Matches: {len(teacher)} ({", ".join(f"{v.name} x{len(s)}" for v, s in plan)})', '',
          match_table_md(report),
          '## Gate A', ''] + [f'- {"PASS" if ok else "FAIL"}: {name}: {value}' for name, value, ok in gate] + [
          '', '## Teacher decisions (share of steps)', '',
          ', '.join(f'{m} {100 * s:.0f}%' for m, s in shares.items()), '',
          'Teacher frags per match: ' + ' '.join('%.0f' % r['frags'] for r in teacher),
          '', 'Teacher deaths per match: ' + ' '.join('%.0f' % r['deaths'] for r in teacher)]
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out + '.md', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(md) + '\n')
    with open(a.out + '.json', 'w') as fh:
        json.dump(dict(cfg={f.name: getattr(cfg, f.name) for f in fields(cfg)}, report=report,
                       mode_share=shares, gate=[(n, v, bool(ok)) for n, v, ok in gate]), fh, indent=1,
                  default=float)
    teacher_figure(report, shares, a.out + '.png')
    print(match_table_md(report))
    for name, value, ok in gate:
        print('PASS' if ok else 'FAIL', name, value)
