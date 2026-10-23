"""The cheater teacher and a random player on the 20 val matches (base graphics), Gate A v2.

    python scripts/dm/dm_teacher_report.py --diag runs/dm/diag_<name>.json [--set name=value ...] [--workers 4]
    ->  reports/dm/e1_teacher.md, reports/dm/e1_teacher.png, reports/dm/e1_teacher.json

--diag is a dm_teacher_diag.py run of the same teacher (death causes and hits by weapon for the figure).
The version 1.1 report (MAP01, Gate A) is at the tag deathmatch-v1.
"""
import argparse
import json
import os
import sys
from dataclasses import fields, replace

import numpy as np

from doomagent.dm.config import DM, MODES, VAL_SEEDS
from doomagent.dm.evaluate import RandomChooser, evaluate_plan, summarize
from doomagent.dm.reports import gate_a2, match_table_md, teacher_figure2
from doomagent.dm.teacher import DMTeacher
from doomagent.variants import BASE


def with_overrides(cfg, items):
    # name=value pairs typed like the DMCfg field
    types = {f.name: type(getattr(cfg, f.name)) for f in fields(cfg)}
    return replace(cfg, **{k: (v == 'True') if types[k] is bool else types[k](v)
                           for k, v in (x.split('=', 1) for x in items or [])})


def mode_share(results):
    modes = np.concatenate([r['chooser']['modes'] for r in results])
    return {m: float((modes == i).mean()) for i, m in enumerate(MODES)}


if __name__ == '__main__':  # worker processes re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--diag', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default='reports/dm/e1_teacher')
    a = ap.parse_args()
    cfg = with_overrides(DM, a.set)
    plan = [(BASE, VAL_SEEDS)]
    teacher = evaluate_plan(plan, DMTeacher(cfg, keep_trace=True), cfg, workers=a.workers, trace=True)
    rand = evaluate_plan(plan, RandomChooser(), cfg, workers=a.workers)
    report = {'teacher': summarize(teacher), 'random': summarize(rand)}
    shares = mode_share(teacher)
    gate = gate_a2(report)
    diag = json.load(open(a.diag))['summary']
    t = report['teacher']
    md = ['# E1: cheater teacher on the val matches (MAP02)', '',
          f'Command: `python {" ".join(sys.argv)}`', '',
          f'Matches: {len(teacher)} (base graphics, VAL_SEEDS)', '',
          match_table_md(report),
          f'Best bot of the match: {t["bot_best"]["mean"]:.2f} [{t["bot_best"]["lo"]:.2f}, {t["bot_best"]["hi"]:.2f}] '
          f'frags; first of 8 in {100 * t["first_share"]:.0f} % of matches.', '',
          '## Gate A v2', ''] + [f'- {"PASS" if ok else "FAIL"}: {name}: {value}' for name, value, ok in gate] + [
          '', '## Teacher decisions (share of steps)', '', ', '.join(f'{m} {100 * s:.0f}%' for m, s in shares.items()),
          '', '## Weapon in hand (share of steps)', '',
          ', '.join(f'{k} {100 * v:.0f}%' for k, v in diag['weapon_share'].items()),
          '', f'## Diagnostics ({a.diag})', '',
          'Deaths by cause: ' + ', '.join(f'{k} {100 * v:.0f}%' for k, v in diag['death_cause'].items()),
          '', 'Hits per shot by weapon: ' + ', '.join(f'{k} {v:.2f} ({diag["shots_by_weapon"][k]} shots)'
                                                     for k, v in diag['hit_share_by_weapon'].items()),
          '', 'Bot on screen, by the reason not to fire: ' +
          ', '.join(f'{k} {100 * v:.0f}%' for k, v in diag['bot_on_screen_by_reason'].items()),
          '', 'Teacher frags per match: ' + ' '.join('%.0f' % r['frags'] for r in teacher),
          '', 'Best bot per match: ' + ' '.join('%.0f' % r['bot_best'] for r in teacher), '',
          '![teacher](e1_teacher.png)', '']
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out + '.md', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(md))
    with open(a.out + '.json', 'w') as fh:
        json.dump(dict(cfg={f.name: getattr(cfg, f.name) for f in fields(cfg)}, report=report, mode_share=shares,
                       gate=[(n, v, bool(ok)) for n, v, ok in gate]), fh, indent=1, default=float)
    teacher_figure2(report, diag, a.out + '.png')
    for name, value, ok in gate:
        print('PASS' if ok else 'FAIL', name, value)
