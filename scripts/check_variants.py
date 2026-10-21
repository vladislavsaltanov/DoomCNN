import json
import os

import cv2
import numpy as np

from doomagent.config import TURN_RIGHT
from doomagent.env import DoomEnv
from doomagent.evaluate import evaluate_plan, summarize
from doomagent.reports import report_table_md
from doomagent.teacher import Teacher
from doomagent.variants import BASE, VARIANTS_DIR, load_variants

SEEDS = (7000, 7001, 7002)  # outside every seed set


def first_frames(v):
    env = DoomEnv(variant=v)
    env.reset(SEEDS[0])
    for _ in range(8):
        env.step(TURN_RIGHT)
    rgb, net = env.screen().copy(), env.obs[-1].copy()
    env.close()
    return rgb, net


base_rgb, base_net = first_frames(BASE)
report = {'base': summarize(evaluate_plan([(BASE, SEEDS)], Teacher()))}
base_reward = report['base']['reward']['mean']
tiles, status = [('base', base_rgb)], []
for v in load_variants(root=VARIANTS_DIR):  # variants already disabled are skipped
    reason = None
    try:
        rgb, net = first_frames(v)
        s = summarize(evaluate_plan([(v, SEEDS)], Teacher()))
        report[v.name] = s
        tiles.append((v.name, rgb))
        diff = float(np.abs(net.astype(int) - base_net.astype(int)).mean())
        if s['reward']['mean'] < 0.7 * base_reward:
            reason = f'teacher reward {s["reward"]["mean"]:.1f} < 0.7 * base'
        elif v.kind in ('tex', 'mix', 'iwad') and diff < 8:
            reason = f'looks like the base graphics (mean diff {diff:.1f})'
    except Exception as e:  # a broken PWAD usually fails in init()
        reason = f'failed: {e}'
    status.append(f'- {v.name} ({v.split}, {v.kind}): ' + (reason or 'ok'))
    if reason:
        path = os.path.join(VARIANTS_DIR, v.name + '.json')
        with open(path) as fh:
            d = json.load(fh)
        d.update(disabled=True, reason=reason)
        with open(path, 'w') as fh:
            json.dump(d, fh, indent=1)

rows = []
for i in range(0, len(tiles), 5):
    row = []
    for name, rgb in tiles[i:i + 5]:
        t = cv2.cvtColor(cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_RGB2BGR)
        cv2.putText(t, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        row.append(t)
    row += [np.zeros_like(row[0])] * (5 - len(row))
    rows.append(np.concatenate(row, axis=1))
os.makedirs('artifacts', exist_ok=True)
os.makedirs('reports', exist_ok=True)
cv2.imwrite('artifacts/s4_variants.png', np.concatenate(rows, axis=0))
with open('reports/s4_variants.md', 'w') as fh:
    fh.write('# S4: graphics variants (teacher on 3 check seeds)\n\n' + report_table_md(report)
             + '\n' + '\n'.join(status) + '\n')
print('\n'.join(status))
