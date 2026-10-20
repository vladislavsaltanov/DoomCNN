import math
import os
from dataclasses import replace

import numpy as np

from doomagent.config import VAL_SEEDS, Cfg
from doomagent.env import DoomEnv
from doomagent.evaluate import evaluate_plan, summarize
from doomagent.teacher import Teacher, enemies_from_labels
from doomagent.variants import BASE

# why the teacher is far from the gate: turn mode and aim margin, then box width against the step
CONFIGS = [
    ('binary, margin 0.2 (default)', Cfg()),
    ('binary, margin 0.1', replace(Cfg(), aim_margin=0.1)),
    ('binary, margin 0.0', replace(Cfg(), aim_margin=0.0)),
    ('delta 3 deg, margin 0.1', replace(Cfg(), turn_mode='delta', turn_delta=0.75, aim_margin=0.1)),
    ('delta 3 deg, margin 0.0', replace(Cfg(), turn_mode='delta', turn_delta=0.75, aim_margin=0.0)),
    ('delta 4 deg, margin 0.2', replace(Cfg(), turn_mode='delta', turn_delta=1.0, aim_margin=0.2)),
    ('delta 4 deg, margin 0.1', replace(Cfg(), turn_mode='delta', turn_delta=1.0, aim_margin=0.1)),
    ('delta 4 deg, margin 0.0', replace(Cfg(), turn_mode='delta', turn_delta=1.0, aim_margin=0.0)),
    ('delta 5 deg, margin 0.1', replace(Cfg(), turn_mode='delta', turn_delta=1.25, aim_margin=0.1)),
]
grid = ['| teacher | reward mean [95% CI] | kills/shot | shots | idle rate | length |',
        '| --- | --- | --- | --- | --- | --- |']
for name, cfg in CONFIGS:
    s = summarize(evaluate_plan([(BASE, VAL_SEEDS[:10])], Teacher(cfg), cfg))
    grid.append(f"| {name} | {s['reward']['mean']:.2f} [{s['reward']['lo']:.2f}, {s['reward']['hi']:.2f}] "
                f"| {s['kills_per_shot']:.3f} | {s['shots']['mean']:.1f} | {s['idle_rate']:.2f} "
                f"| {s['length']['mean']:.0f} |")

cfg = Cfg()
env, teacher, widths = DoomEnv(cfg, BASE), Teacher(cfg), []
for seed in VAL_SEEDS[:10]:
    obs = env.reset(seed)
    teacher.reset()
    done = False
    while not done:
        priv = env.privileged()
        widths += [e.width_px for e in enemies_from_labels(priv.labels, priv.px, priv.py, cfg.width)]
        obs, _, done = env.step(teacher(obs, priv))
env.close()
w = np.asarray(widths)
box = ['| turn step | narrowest box that can hold the crosshair | share of enemy boxes at least that wide |',
       '| --- | --- | --- |']
for name, step_deg in (('binary, 14.06 deg', 14.06), ('delta 8 deg', 8.0), ('delta 4 deg', 4.0)):
    need = math.tan(math.radians(step_deg)) * cfg.width / 2  # one step, in network pixels
    box.append(f'| {name} | {need:.1f} px | {np.mean(w >= need) * 100:.1f}% |')
os.makedirs('reports', exist_ok=True)
with open('reports/s1_turn_grid.md', 'w') as fh:
    fh.write('# S1: turn mode and aim margin, first 10 val seeds (base graphics)\n\n'
             + '\n'.join(grid) + '\n\n'
             + f'# S1: enemy box width vs turn step ({len(w)} enemy-box samples, '
             + f'median {np.median(w):.1f} px at 160 px width)\n\n' + '\n'.join(box) + '\n')
print('\n'.join(grid))
print()
print('\n'.join(box))
