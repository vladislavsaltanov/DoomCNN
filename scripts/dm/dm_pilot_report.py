"""E2 pilot report: reports/dm/e2_pilot.md and reports/dm/e2_pilot.png from the two pilot runs (four heads).

    python scripts/dm/dm_pilot_report.py
"""
import json

import numpy as np

from doomagent.dm.actions import HEADS
from doomagent.dm.reports import curves_figure

RUNS = {'m0': 'runs/dm/m0_pilot.pt', 'm1': 'runs/dm/m1_pilot.pt'}


def load(p):
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


meta = load('data/dm/pilot/meta.json')
teacher = load('reports/dm/e1_teacher.json')['report']['teacher']['frags']
curves_figure({k: v + '.log.json' for k, v in RUNS.items()}, 'reports/dm/e2_pilot.png', teacher=teacher,
              confusion_of=RUNS['m1'] + '.live.json', head='turn')
lines = ['# E2: pilot M0 and M1 with four heads', '',
         f'Data: data/dm/pilot, {meta["matches"]} teacher matches, {meta["steps"]} steps '
         f'(5% of matches held out). Validation: 10 base matches (VAL_SEEDS[:10]) every 5 epochs. '
         f'Teacher on the 20 val seeds (E1): {teacher["mean"]:.2f} [{teacher["lo"]:.2f}, {teacher["hi"]:.2f}] frags.', '',
         '| run | epochs | minutes | held-out CE first → last (min) | best val frags (epoch) | stuck at best |',
         '|---|---|---|---|---|---|']
heads = {}
for k, path in RUNS.items():
    log = load(path + '.log.json')['log']
    live = load(path + '.live.json')['holdout_confusion']
    best = max(log, key=lambda r: r['val_score'])
    f = best['frags']
    lines.append(f'| {k.upper()} | {log[-1]["epoch"]} | {log[-1]["minutes"]:.1f} | '
                 f'{log[0]["holdout_ce"]:.2f} → {log[-1]["holdout_ce"]:.2f} ({min(r["holdout_ce"] for r in log):.2f}) | '
                 f'{f["mean"]:.1f} [{f["lo"]:.1f}, {f["hi"]:.1f}] ({best["epoch"]}) | {best["stuck_share"]:.0%} |')
    for h in HEADS:
        conf = np.array(live[h], float)
        heads.setdefault(h, {})[k] = (max(r[f'holdout_acc_{h}'] for r in log), conf.sum(1).max() / max(conf.sum(), 1))
lines += ['', 'Held-out accuracy per head (best epoch) against the share of its most frequent label:', '',
          '| head | ' + ' | '.join(f'{k.upper()} accuracy' for k in RUNS) + ' | most frequent label |',
          '|---|' + '---|' * (len(RUNS) + 1)]
for h in HEADS:
    lines.append(f'| {h} | ' + ' | '.join(f'{heads[h][k][0]:.2f}' for k in RUNS) + f' | {heads[h]["m0"][1]:.2f} |')
lines += ['', '![curves](e2_pilot.png)', '']
with open('reports/dm/e2_pilot.md', 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines))
print('written reports/dm/e2_pilot.md, reports/dm/e2_pilot.png')  # the table has arrows a cp1251 console cannot print
