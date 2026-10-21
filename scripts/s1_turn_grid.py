import math
import os
from dataclasses import replace

import numpy as np

from doomagent.config import VAL_SEEDS, Cfg
from doomagent.env import DoomEnv
from doomagent.evaluate import evaluate_plan, summarize
from doomagent.teacher import Teacher, enemies_from_labels
from doomagent.variants import BASE

# five actions: coarse turns scan, fine turns aim. Fine values are degrees per step (4 tics).
CONFIGS = [
    ('binary, fine 4 deg, zone 0.15, margin 0.2', replace(Cfg(), fine_turn_delta=1.0)),
    ('binary, fine 2 deg, zone 0.15, margin 0.2', replace(Cfg(), fine_turn_delta=0.5)),
    ('binary, fine 1 deg, zone 0.15, margin 0.2', replace(Cfg(), fine_turn_delta=0.25)),
    ('delta 16 deg, fine 4 deg, zone 0.15, margin 0.2',
     replace(Cfg(), turn_mode='delta', fine_turn_delta=1.0)),
    ('delta 16 deg, fine 2 deg, zone 0.15, margin 0.2',
     replace(Cfg(), turn_mode='delta', fine_turn_delta=0.5)),
    ('binary, fine 2 deg, zone 0.10, margin 0.2',
     replace(Cfg(), fine_turn_delta=0.5, fine_zone=0.10)),
    ('binary, fine 2 deg, zone 0.25, margin 0.2',
     replace(Cfg(), fine_turn_delta=0.5, fine_zone=0.25)),
    ('binary, fine 2 deg, zone 0.15, margin 0.0',
     replace(Cfg(), fine_turn_delta=0.5, aim_margin=0.0)),
    ('delta 16 deg, fine 2 deg, zone 0.15, margin 0.0',
     replace(Cfg(), turn_mode='delta', fine_turn_delta=0.5, aim_margin=0.0)),
    ('delta 16 deg, fine 2 deg, zone 0.25, margin 0.2',
     replace(Cfg(), turn_mode='delta', fine_turn_delta=0.5, fine_zone=0.25)),
]
grid = ['| teacher | reward mean [95% CI] | kills/shot | shots | idle rate | length |',
        '| --- | --- | --- | --- | --- | --- |']
for name, cfg in CONFIGS:
    s = summarize(evaluate_plan([(BASE, VAL_SEEDS[:10])], Teacher(cfg), cfg))
    grid.append(f"| {name} | {s['reward']['mean']:.2f} [{s['reward']['lo']:.2f}, "
                f"{s['reward']['hi']:.2f}] | {s['kills_per_shot']:.3f} | {s['shots']['mean']:.1f} "
                f"| {s['idle_rate']:.2f} | {s['length']['mean']:.0f} |")

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
for name, step_deg in (('coarse binary, 14.06 deg', 14.06), ('coarse delta, 16 deg', 16.0),
                       ('fine, 4 deg', 4.0), ('fine, 2 deg', 2.0), ('fine, 1 deg', 1.0)):
    need = math.tan(math.radians(step_deg)) * cfg.width / 2  # one step, in network pixels
    box.append(f'| {name} | {need:.1f} px | {np.mean(w >= need) * 100:.1f}% |')
os.makedirs('reports', exist_ok=True)
with open('reports/s1_turn_grid.md', 'w') as fh:
    fh.write('# S1: five actions, fine turn size, fine zone and aim margin '
             + '(first 10 val seeds, base graphics)\n\n'
             + '\n'.join(grid) + '\n\n'
             + f'# S1: enemy box width vs turn step ({len(w)} enemy-box samples, '
             + f'median {np.median(w):.1f} px at 160 px width)\n\n' + '\n'.join(box) + '\n')
print('\n'.join(grid))
print()
print('\n'.join(box))
