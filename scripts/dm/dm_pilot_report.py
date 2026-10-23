"""D2 pilot report: reports/dm/d2_pilot.md and reports/dm/d2_pilot.png from the two pilot runs.

    python scripts/dm/dm_pilot_report.py
"""
import json

import numpy as np

from doomagent.dm.config import ACTION_NAMES
from doomagent.dm.reports import curves_figure

RUNS = {'m0': 'runs/dm/m0_pilot.pt', 'm1': 'runs/dm/m1_pilot.pt'}


def load(p):
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


meta = load('data/dm/pilot/meta.json')
teacher = load('reports/dm/d1_teacher.json')['report']['teacher']['frags']
curves_figure({k: v + '.log.json' for k, v in RUNS.items()}, 'reports/dm/d2_pilot.png', teacher=teacher,
              confusion_of=RUNS['m1'] + '.live.json', action_names=ACTION_NAMES)
lines = ['# D2: pilot M0 and M1', '',
         f'Data: data/dm/pilot, {meta["matches"]} teacher matches, {meta["steps"]} steps '
         f'(5% of matches held out). Validation: 10 base matches (VAL_SEEDS[:10]) every 5 epochs. '
         f'Teacher on the 20 val seeds (D1): {teacher["mean"]:.2f} [{teacher["lo"]:.2f}, {teacher["hi"]:.2f}] frags.', '',
         '| run | epochs | minutes | held-out CE first → last | train acc last | held-out acc best | most frequent action | best val frags (epoch) | stuck at best |',
         '|---|---|---|---|---|---|---|---|---|']
per = {}
for k, path in RUNS.items():
    log = load(path + '.log.json')['log']
    conf = np.array(load(path + '.live.json')['holdout_confusion'], float)
    majority = conf.sum(1).max() / conf.sum()
    best = max(log, key=lambda r: r['val_score'])
    f = best['frags']
    lines.append(f'| {k.upper()} | {log[-1]["epoch"]} | {log[-1]["minutes"]:.1f} | '
                 f'{log[0]["holdout_ce"]:.2f} → {log[-1]["holdout_ce"]:.2f} (min {min(r["holdout_ce"] for r in log):.2f}) | '
                 f'{log[-1]["train_acc"]:.2f} | {max(r["holdout_acc"] for r in log):.2f} | {majority:.2f} | '
                 f'{f["mean"]:.1f} [{f["lo"]:.1f}, {f["hi"]:.1f}] ({best["epoch"]}) | {best["stuck_share"]:.0%} |')
    per[k] = log[-1]['holdout_acc_per_action']
lines += ['', 'Held-out accuracy per action, last epoch (chance 0.10):', '',
          '| action | ' + ' | '.join(k.upper() for k in per) + ' |', '|---|' + '---|' * len(per)]
for i, name in enumerate(ACTION_NAMES):
    lines.append(f'| {name} | ' + ' | '.join('—' if per[k][i] is None else f'{per[k][i]:.2f}' for k in per) + ' |')
lines += ['', '![curves](d2_pilot.png)', '']
with open('reports/dm/d2_pilot.md', 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines))
print('written reports/dm/d2_pilot.md, reports/dm/d2_pilot.png')  # the table has arrows a cp1251 console cannot print
