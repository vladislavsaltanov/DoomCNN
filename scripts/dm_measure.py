"""D0 measurements on cig MAP01: physics per step, number of bots, labels and items, map specials,
repeatability and distinct seeds.

    python scripts/dm_measure.py   ->  reports/d0_env.md, reports/d0_env.png
"""
import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import replace

import numpy as np
import vizdoom as vzd

from doomagent.dm.config import (BOT, DM, FINE_LEFT, FINE_RIGHT, FORWARD, FORWARD_LEFT,
                                 FORWARD_RIGHT, ITEM_KIND, N_ACTIONS, STRAFE_LEFT, TURN_LEFT,
                                 TURN_RIGHT, VAL_SEEDS)
from doomagent.dm.env import DeathmatchEnv
from doomagent.dm.evaluate import run_match, summarize
from doomagent.dm.reports import d0_figure
from doomagent.evaluate import evaluate_plan
from doomagent.variants import BASE
from doomagent.wadgen import read_wad

GV = vzd.GameVariable
STEPS = 12


class RandomAgent:
    # uniform over the ten actions, the same sequence in every match; picklable for workers
    def reset(self):
        self.rng = np.random.default_rng(0)

    def __call__(self, obs, priv):
        return int(self.rng.integers(0, N_ACTIONS))


def pose(env):
    g = env.game
    return g.get_game_variable(GV.POSITION_X), g.get_game_variable(GV.POSITION_Y), \
        g.get_game_variable(GV.ANGLE)


def idle(env, steps=8):
    # stand still between measurements: no agent action does that
    for _ in range(steps):
        env.game.make_action([0] * 7, 4)


def moves(env, actions):
    out = []
    for a in actions:
        x, y, ang = pose(env)
        env.step(a)
        x2, y2, ang2 = pose(env)
        out.append((round(math.hypot(x2 - x, y2 - y), 1), round((ang2 - ang + 180) % 360 - 180, 2)))
    return out


def physics():
    # a match without bots; straight runs from 5 spawn spots, the longest free run kept
    env = DeathmatchEnv(replace(DM, bots=0))
    out = {}
    for seed in VAL_SEEDS[:5]:
        env.reset(seed)
        idle(env)
        for key, action in (('forward', FORWARD), ('strafe_left', STRAFE_LEFT)):
            run = [d for d, _ in moves(env, [action] * STEPS)]
            idle(env)
            if sum(run) > sum(out.get(key, [])):
                out[key] = run
    env.reset(VAL_SEEDS[0])
    idle(env)
    for key, a, b in (('turn_left', TURN_LEFT, TURN_RIGHT), ('fine_left', FINE_LEFT, FINE_RIGHT),
                      ('forward_left', FORWARD_LEFT, FORWARD_RIGHT)):
        hold = moves(env, [a] * STEPS)
        idle(env)
        alternate = moves(env, [a, b] * (STEPS // 2))
        idle(env)
        out[key] = dict(hold=[t for _, t in hold], alternate=[t for _, t in alternate],
                        hold_dist=[d for d, _ in hold])
    env.close()
    return out


def bot_count(n, workers):
    cfg = replace(DM, bots=n)
    env = DeathmatchEnv(cfg)
    t0 = time.perf_counter()
    r = run_match(env, RandomAgent(), VAL_SEEDS[0])
    ms = 1000.0 * (time.perf_counter() - t0) / r['length']  # a fresh game included
    env.close()
    res = evaluate_plan([(BASE, VAL_SEEDS[:5])], RandomAgent(), cfg, workers=workers)
    s = summarize(res)
    bots = np.array([f for x in res for f in x['bot_frags']])
    return dict(ms_per_step=ms, bot_median=s['bot_median'], bot_mean=float(bots.mean()),
                bot_negative=float((bots < 0).mean()),
                match_frags=float(np.mean([sum(x['bot_frags']) + x['frags'] for x in res])),
                random_frags=s['frags'], random_suicides=s['suicides'],
                random_enemy_share=s['enemy_share']['mean'])


def labels_and_items():
    env = DeathmatchEnv(DM)
    agent, names = RandomAgent(), Counter()
    agent.reset()
    obs, done = env.reset(VAL_SEEDS[1]), False
    start = Counter(o.name for o in env.privileged().objects)
    while not done:
        for lb in env.privileged().labels:
            names[lb.object_name] += 1
        obs, _, done = env.step(agent(obs, env.privileged()))
    end = Counter(o.name for o in env.privileged().objects)
    picked = env.privileged().items
    env.close()
    return names, {k: (start[k], end[k]) for k in sorted(set(start) | set(end))}, picked


def map_specials():
    lumps = read_wad(os.path.join(vzd.scenarios_path, 'cig.wad'))
    names = [n for n, _ in lumps]
    textmap = lumps[names.index('MAP01') + 1][1].decode('ascii', 'replace')
    return dict(Counter(re.findall(r'special\s*=\s*(\d+)', textmap)))


def signature(env, seed, steps):
    env.reset(seed)
    h = hashlib.md5()
    for _ in range(steps):
        obs, _, _ = env.step(FORWARD)
        h.update(obs.frames[-1].tobytes())
    bots = sorted((round(o.x), round(o.y)) for o in env.privileged().objects if o.name == BOT)
    return h.hexdigest(), tuple(bots)


def repeats():
    cfg = replace(DM, timelimit=0.25)
    sigs = []
    for _ in range(2):
        env = DeathmatchEnv(cfg)
        sigs.append(signature(env, VAL_SEEDS[2], 120))
        env.close()
    env = DeathmatchEnv(cfg)
    distinct = len({signature(env, s, 3)[1] for s in VAL_SEEDS})
    env.close()
    return sigs[0] == sigs[1], distinct


def report_md(phys, bots, names, items, picked, specials, same, distinct):
    md = ['# D0: deathmatch environment on cig.wad MAP01', '',
          '## Label names (random agent, one 2-minute match, label-steps)', '',
          '| name | class | label-steps |', '| --- | --- | --- |']
    for n, c in names.most_common():
        cls = 'bot' if n == BOT else ITEM_KIND.get(n, 'other')
        md.append(f'| {n} | {cls} | {c} |')
    md += ['', '## Physics per agent step (4 tics), no bots', '',
           '| action | per step |', '| --- | --- |',
           '| forward, held (units) | ' + ' '.join('%.1f' % v for v in phys['forward']) + ' |',
           '| strafe left, held (units) | ' + ' '.join('%.1f' % v for v in phys['strafe_left']) + ' |']
    for key in ('turn_left', 'fine_left', 'forward_left'):
        for mode in ('hold', 'alternate'):
            md.append(f'| {key.replace("_", " ")}, {mode} (degrees) | '
                      + ' '.join('%.2f' % v for v in phys[key][mode]) + ' |')
    md.append('| forward left, held (units) | '
              + ' '.join('%.1f' % v for v in phys['forward_left']['hold_dist']) + ' |')
    md += ['', '## Number of bots (random agent, 5 val matches each)', '',
           '| bots | ms per step | median bot frags [95% CI] | mean bot frags | bots below 0 | '
           'all frags per match | random agent frags | random suicides | random sees a bot |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for n, b in sorted(bots.items()):
        m = b['bot_median']
        md.append(f'| {n} | {b["ms_per_step"]:.1f} | {m["mean"]:.2f} [{m["lo"]:.2f}, {m["hi"]:.2f}] | '
                  f'{b["bot_mean"]:.2f} | {b["bot_negative"]:.0%} | {b["match_frags"]:.1f} | '
                  f'{b["random_frags"]["mean"]:.1f} | {b["random_suicides"]["mean"]:.1f} | '
                  f'{b["random_enemy_share"]:.0%} |')
    md += ['', '## Items and map', '',
           '- objects at match start and end (objects_info): '
           + ', '.join(f'{k} {a}/{b}' for k, (a, b) in items.items()),
           f'- items the random agent picked up: {picked:.0f}',
           f'- MAP01 line specials: {json.dumps(specials)} (no doors or lifts: no USE button)',
           '', '## Repeatability', '',
           f'- two matches with one seed and one action sequence (120 steps): '
           f'{"identical" if same else "DIFFERENT"}',
           f'- distinct matches among the {len(VAL_SEEDS)} VAL_SEEDS: {distinct}']
    return '\n'.join(md) + '\n'


if __name__ == '__main__':  # worker processes re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default='reports/d0_env')
    a = ap.parse_args()
    phys = physics()
    print('physics', json.dumps(phys))
    bots = {n: bot_count(n, a.workers) for n in (3, 5, 7)}
    for n, b in bots.items():
        print('bots', n, json.dumps({k: v for k, v in b.items() if k != 'random_frags'}, default=str))
    names, items, picked = labels_and_items()
    specials = map_specials()
    same, distinct = repeats()
    print('repeat:', 'identical' if same else 'DIFFERENT', '| distinct val matches:', distinct)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out + '.md', 'w', encoding='utf-8') as fh:
        fh.write(report_md(phys, bots, names, items, picked, specials, same, distinct))
    d0_figure(phys, bots, a.out + '.png')
    print('written', a.out + '.md', a.out + '.png')
