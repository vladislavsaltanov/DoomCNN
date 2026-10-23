"""E0 measurements on cig MAP02: turns and moves per step, the next-weapon key and auto-switch on pickup,
projectile speeds, map specials (doors, lifts), repeatability, time per step and bot strength.

    python scripts/dm/dm_measure.py   ->  reports/dm/e0_env.md, reports/dm/e0_env.png, reports/dm/e0_env.json

The D0 version for MAP01 is at the tag deathmatch-v1.
"""
import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np
import vizdoom as vzd

from doomagent.dm.actions import FORWARD, MOVE_NAMES, NOOP, STAY, TURN_DEG, turn_index
from doomagent.dm.config import DM, PROJECTILES, VAL_SEEDS, WEAPON_ITEMS
from doomagent.dm.env import DeathmatchEnv
from doomagent.dm.evaluate import RandomChooser, evaluate_plan, summarize
from doomagent.dm.nav import lookahead, nav_for
from doomagent.dm.reports import e0_figure
from doomagent.variants import BASE
from doomagent.wadgen import read_wad

GV = vzd.GameVariable
WARM = 6  # steps a player who just teleported in cannot move


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def bearing(px, py, angle, x, y):
    # degrees from the view direction to (x, y), + is left
    return wrap(math.degrees(math.atan2(y - py, x - px)) - angle)


def turns(env, seed):
    out = {}
    for i in range(1, len(TURN_DEG)):
        env.reset(seed)
        for _ in range(WARM):
            env.step(NOOP)
        a0 = env.privileged().angle
        env.step((STAY, i, 0, 0))
        out[TURN_DEG[i]] = round(wrap(env.privileged().angle - a0), 3)
    return out


def moves(env, seeds, steps=8):
    # map units per step, mean of the last 4 of `steps` (full speed); a wall next to a spawn spot cuts a run
    # short, so the best of several spawn spots counts
    out = {}
    for m in range(1, len(MOVE_NAMES)):
        best = 0.0
        for seed in seeds:
            env.reset(seed)
            for _ in range(WARM):
                env.step(NOOP)
            d = []
            for _ in range(steps):
                p = env.privileged()
                env.step((m, 0, 0, 0))
                q = env.privileged()
                d.append(math.hypot(q.px - p.px, q.py - p.py))
            best = max(best, float(np.mean(d[-4:])))
        out[MOVE_NAMES[m]] = round(best, 1)
    return out


def walk_to_weapon(env, seed, max_steps=400):
    # A* walk to the nearest weapon on the map; records the weapon in hand around the pickup
    env.reset(seed)
    p = env.privileged()
    nav = nav_for(p.sectors)
    weapons = [o for o in p.objects if o.name in WEAPON_ITEMS]
    goal = min(weapons, key=lambda o: math.hypot(o.x - p.px, o.y - p.py))
    owned0 = p.owned
    for t in range(max_steps):
        p = env.privileged()
        if p.owned != owned0:
            slot = next(i for i in range(len(p.owned)) if p.owned[i] > owned0[i])
            after = []
            for _ in range(6):
                env.step(NOOP)
                after.append(env.privileged().weapon)
            return dict(seed=seed, weapon=goal.name, slot=slot, steps=t, weapon_after=after,
                        auto_switch=slot in after)
        path = nav.path((p.px, p.py), (goal.x, goal.y))
        if not path:
            return dict(seed=seed, weapon=goal.name, error='no path')
        x, y = lookahead(path, p.px, p.py)
        rel = bearing(p.px, p.py, p.angle, x, y)
        turn = min(range(len(TURN_DEG)), key=lambda i: abs(rel - TURN_DEG[i]))
        _, _, done = env.step((FORWARD if abs(rel) < 30 else STAY, turn, 0, 0))
        if done or p.deaths != env.privileged().deaths:
            return dict(seed=seed, weapon=goal.name, error='died or match over')
    return dict(seed=seed, weapon=goal.name, error='not reached')


def projectile_speeds(env, seeds):
    # map units per tic of each projectile class, from object positions on consecutive steps (objects: a
    # rocket flies 80 units a step and is rarely on screen twice; the measurement only, never the teacher)
    speeds = defaultdict(list)
    chooser = RandomChooser()
    for seed in seeds:
        chooser.reset()
        obs, done, prev = env.reset(seed), False, {}
        while not done:
            p = env.privileged()
            now = {o.id: (o.name, o.x, o.y) for o in p.objects if o.name in PROJECTILES}
            for k, (name, x, y) in now.items():
                if k in prev:
                    d = math.hypot(x - prev[k][1], y - prev[k][2]) / DM.frame_skip
                    if d > 1.0:  # not just spawned or exploding
                        speeds[name].append(d)
            prev = now
            obs, _, done = env.step(chooser(obs, p))
    return {k: dict(median=round(float(np.median(v)), 2), n=len(v)) for k, v in speeds.items()}


def specials():
    # line specials of MAP02 in TEXTMAP: doors and lifts would need the use key
    wad = read_wad(os.path.join(vzd.scenarios_path, 'cig.wad'))
    i = [n for n, _ in wad].index(DM.map.upper())
    text = wad[i + 1][1].decode('ascii', 'replace')
    lines = re.findall(r'linedef\s*\{([^}]*)\}', text, re.I)
    found = Counter(int(m) for body in lines for m in re.findall(r'special\s*=\s*(\d+)', body))
    return dict(linedefs=len(lines), specials=dict(found))


def repeat(seed):
    def run():
        e, c = DeathmatchEnv(replace(DM, timelimit=0.2)), RandomChooser()
        c.reset()
        obs, done, h = e.reset(seed), False, hashlib.md5()
        while not done:
            h.update(obs.frames[-1].tobytes())
            obs, _, done = e.step(c(obs, e.privileged()))
        e.close()
        return h.hexdigest()
    return run() == run()


if __name__ == '__main__':  # evaluation workers re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument('--matches', type=int, default=5)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    env = DeathmatchEnv(DM)
    t_turn = turns(env, VAL_SEEDS[0])
    t_move = moves(env, VAL_SEEDS[:6])
    walks = [walk_to_weapon(env, s) for s in VAL_SEEDS[:6]]
    speeds = projectile_speeds(env, VAL_SEEDS[:2])
    t0 = time.perf_counter()
    env.reset(VAL_SEEDS[1])
    for i in range(300):
        env.step((FORWARD, turn_index(3.0) if i % 2 else 0, 0, 0))
    ms = (time.perf_counter() - t0) / 300 * 1000
    env.close()
    res = evaluate_plan([(BASE, VAL_SEEDS[:a.matches])], RandomChooser(), DM, workers=a.workers)
    s = summarize(res)
    bots = dict(player=s['frags'], bot_median=s['bot_median'], bot_best=s['bot_best'])
    out = dict(turns=t_turn, moves=t_move, weapon_walks=walks, projectiles=speeds, map=specials(),
               repeat=repeat(VAL_SEEDS[2]), ms_per_step=round(ms, 2), bots=bots,
               bot_frags=[r['bot_frags'] for r in res])
    os.makedirs('reports/dm', exist_ok=True)
    with open('reports/dm/e0_env.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    e0_figure(t_turn, t_move, bots, 'reports/dm/e0_env.png')
    worst = max(abs(v - k) for k, v in t_turn.items())
    ok_walks = [w for w in walks if 'auto_switch' in w]
    ci = lambda d: f"{d['mean']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"  # noqa: E731
    lines = ['# E0: cig MAP02 environment', '',
             f'Turns (requested → executed, degrees): worst error {worst:.3f}. ' +
             ', '.join(f'{k:+g} → {v:+.2f}' for k, v in sorted(t_turn.items())), '',
             'Moves at full speed (map units per step): ' + ', '.join(f'{k} {v}' for k, v in t_move.items()), '',
             f'Walks to the nearest weapon (A*): {len(ok_walks)} of {len(walks)} reached; auto-switch on pickup in '
             f'{sum(w["auto_switch"] for w in ok_walks)} of {len(ok_walks)}. ' +
             '; '.join(f'seed {w["seed"]} {w["weapon"]}: ' + (f'slot {w["slot"]} after {w["steps"]} steps, '
                                                             f'in hand {w["weapon_after"]}' if 'slot' in w else w['error'])
                       for w in walks), '',
             'Projectile speeds (map units per tic, median over object tracks): ' +
             ', '.join(f'{k} {v["median"]} (n={v["n"]})' for k, v in speeds.items()), '',
             f'Map {DM.map.upper()}: {out["map"]["linedefs"]} linedefs, specials {out["map"]["specials"]}', '',
             f'Same seed twice, same random actions: frames identical = {out["repeat"]}; {out["ms_per_step"]} ms per step.', '',
             f'Random player on VAL_SEEDS[:{a.matches}]: player {ci(bots["player"])}, median bot '
             f'{ci(bots["bot_median"])}, best bot {ci(bots["bot_best"])} frags per match.', '',
             '![e0](e0_env.png)', '']
    with open('reports/dm/e0_env.md', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print('written reports/dm/e0_env.{md,png,json}; worst turn error %.3f' % worst)
