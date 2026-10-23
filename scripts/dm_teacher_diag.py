"""Where the deathmatch teacher loses: deaths, shots and fights on the val matches.

    python scripts/dm_teacher_diag.py [--matches 20] [--workers 2] [--set key=value ...]

Per death: a suicide (frags went down), a spawn death (life shorter than 2 s), whether a bot was on
screen in the 2 s before, the teacher's mode and health 2 s before. Per shot: did the hit counter
rise within 1.5 s, the distance and width of the target. Per fight (a run of steps with a bot on
screen): steps to the first shot, kill or death within it. Plus the teacher's rank among the 8
players. Writes reports/d3_teacher_diag.{md,json}.
"""
import argparse
import json
import math
from collections import Counter
from dataclasses import replace

import numpy as np

from doomagent.dm.config import DM, FRESH_SEEDS, MODES, SHOOT, VAL_SEEDS
from doomagent.dm.teacher import visible_bots

WINDOW = 17      # 2 s of agent steps
HIT_WINDOW = 13  # 1.5 s: a rocket at 20 units per tic flies about 1000 units


WHY = ('shoot', 'no_combat', 'not_aimed', 'reloading', 'too_close', 'wall', 'barrel')


def why_no_shot(teacher, p, a, cfg):
    # the first of the teacher's shot conditions that failed on this step
    from doomagent.dm.teacher import Enemy, is_aimed
    if a == SHOOT:
        return 'shoot'
    t = teacher.target
    if teacher.mode != 'combat' or t is None:
        return 'no_combat'
    slack, lead = cfg.aim_slack / max(t.dist, 1.0), teacher.lead or 0.0
    aim = Enemy(t.id, t.left + lead - slack, t.right + lead + slack, t.dist, t.width_px)
    if not is_aimed(aim, cfg.aim_margin):
        return 'not_aimed'
    if not p.attack_ready:
        return 'reloading'
    if t.dist < cfg.safe_dist:
        return 'too_close'
    ray = teacher.nav.shot_dist if cfg.shot_check == 'rocket' else teacher.nav.wall_dist
    if ray(p.px, p.py, p.angle) < cfg.safe_dist:
        return 'wall'
    return 'barrel'


def run_one(seed, cfg):
    from doomagent.dm.env import DeathmatchEnv
    from doomagent.dm.teacher import DMTeacher
    env, teacher = DeathmatchEnv(cfg), DMTeacher(cfg)
    obs, done, rows = env.reset(seed), False, []
    while not done:
        p = env.privileged()
        a = teacher(obs, p)
        bots = visible_bots(p.labels, p.px, p.py, cfg.width)
        near = min(bots, key=lambda e: e.dist) if bots else None
        t = teacher.target if teacher.mode == 'combat' else None
        rows.append(dict(a=int(a), why=why_no_shot(teacher, p, a, cfg), mode=MODES.index(teacher.mode),
                         off=t.offset if t else math.nan, half=(t.right - t.left) / 2 if t else math.nan,
                         lead=(teacher.lead or 0.0) if t else math.nan, tdist=t.dist if t else math.nan,
                         health=p.health, ammo=p.ammo, frags=p.frags,
                         deaths=p.deaths, hits=p.hits, bot=bool(bots), n_bots=len(bots),
                         dist=near.dist if near else math.nan, width=near.width_px if near else math.nan,
                         first=bool(obs.first)))
        obs, _, done = env.step(a)
    end = env.privileged()
    env.close()
    return dict(seed=seed, rows=rows, player_frags=[float(f) for f in end.player_frags], frags=end.frags,
                deaths=end.deaths)


def analyze(m):
    rows = m['rows']
    T = len(rows)
    col = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    deaths, shots, fights = [], [], []
    life_start = 0
    for t in range(1, T):
        if col['deaths'][t] > col['deaths'][t - 1]:
            w0 = max(life_start, t - WINDOW)
            deaths.append(dict(suicide=bool(col['frags'][t] < col['frags'][t - 1]),
                               spawn=t - life_start < WINDOW, bot_seen=bool(col['bot'][w0:t].any()),
                               mode=MODES[int(col['mode'][t - 1])], health_before=float(col['health'][w0]),
                               shots_before=int((col['a'][w0:t] == SHOOT).sum())))
        if col['first'][t]:
            life_start = t
    for t in np.where(col['ammo'][1:] < col['ammo'][:-1])[0]:
        if col['first'][t + 1]:
            continue  # a respawn refill, not a shot
        hit = bool(col['hits'][min(t + HIT_WINDOW, T - 1)] > col['hits'][t])
        shots.append(dict(hit=hit, dist=float(col['dist'][t]), width=float(col['width'][t])))
    t = 0
    while t < T:
        if not col['bot'][t]:
            t += 1
            continue
        s = t
        while t < T and col['bot'][t] and not col['first'][t]:
            t += 1
        e = min(t + 4, T - 1)
        shot_at = np.where(col['a'][s:t] == SHOOT)[0]
        fights.append(dict(steps=t - s, first_shot=int(shot_at[0]) if len(shot_at) else None,
                           kill=bool(col['frags'][e] > col['frags'][s]), died=bool(col['deaths'][e] > col['deaths'][s]),
                           close=float(np.nanmin(col['dist'][s:t])) if t > s else math.nan,
                           health=float(col['health'][s])))
        t += 1
    pf = m['player_frags']
    rank = 1 + sum(f > pf[0] for f in pf[1:])
    na = [r for r in rows if r['why'] == 'not_aimed']
    if na:
        off = np.array([r['off'] for r in na])
        lead = np.array([r['lead'] for r in na])
        half = np.array([r['half'] for r in na])
        aim_err = np.abs(off + lead)  # the aim point's distance from the screen center
        lead_jump = [abs(b['lead'] - a['lead']) for a, b in zip(rows, rows[1:])
                     if a['why'] != 'no_combat' and b['why'] != 'no_combat' and not math.isnan(a['lead'] + b['lead'])]
        aim_stats = dict(err_median=float(np.median(aim_err)), half_median=float(np.median(half)),
                         lead_abs_median=float(np.median(np.abs(lead))), lead_jump_median=float(np.median(lead_jump)) if lead_jump else None,
                         err_in_fine_zone=float(np.mean(aim_err <= DM.fine_zone)),
                         err_le_half_if_no_lead=float(np.mean(np.abs(off) <= half)),
                         dist_median=float(np.nanmedian([r['tdist'] for r in na])),
                         actions=dict(Counter(int(r['a']) for r in na)))
    else:
        aim_stats = {}
    combat = [r['why'] for r in rows if r['bot']]
    why = {w: sum(x == w for x in combat) for w in WHY}
    return dict(deaths=deaths, shots=shots, fights=fights, rank=rank, best_bot=max(pf[1:]), why=why, aim=aim_stats,
                frags=m['frags'], n_deaths=m['deaths'], steps=T,
                low_health_combat=float(((col['mode'] == 0) & (col['health'] < 40)).mean()),
                close_no_shot=float(((col['bot']) & (col['dist'] < 128)).mean()))


def summary(res):
    d = [x for r in res for x in r['deaths']]
    s = [x for r in res for x in r['shots']]
    f = [x for r in res for x in r['fights']]
    n = max(len(d), 1)
    by_mode = {m: sum(x['mode'] == m for x in d) / n for m in MODES}
    dist = np.array([x['dist'] for x in s])
    bins = [(0, 256), (256, 512), (512, 1024), (1024, 1e9)]
    hit_by_dist = {f'{lo}-{hi if hi < 1e9 else "inf"}': (float(np.mean([x['hit'] for x, dd in zip(s, dist) if lo <= dd < hi]))
                                                       if any(lo <= dd < hi for dd in dist) else None,
                                                       int(sum(lo <= dd < hi for dd in dist))) for lo, hi in bins}
    fs = [x for x in f if x['steps'] >= 3]
    first = [x['first_shot'] for x in fs if x['first_shot'] is not None]
    why = {w: sum(r['why'][w] for r in res) for w in WHY}
    nw = max(sum(why.values()), 1)
    return dict(
        bot_on_screen_steps_by_reason={w: v / nw for w, v in why.items()},
        matches=len(res), frags=float(np.mean([r['frags'] for r in res])), deaths=float(np.mean([r['n_deaths'] for r in res])),
        rank_mean=float(np.mean([r['rank'] for r in res])), rank_first_share=float(np.mean([r['rank'] == 1 for r in res])),
        best_bot_mean=float(np.mean([r['best_bot'] for r in res])),
        deaths_total=len(d), suicide_share=sum(x['suicide'] for x in d) / n, spawn_share=sum(x['spawn'] for x in d) / n,
        unseen_share=sum(not x['bot_seen'] for x in d) / n, death_mode_share=by_mode,
        death_health_before=float(np.mean([x['health_before'] for x in d])) if d else None,
        shots_per_match=len(s) / max(len(res), 1), hit_share=float(np.mean([x['hit'] for x in s])) if s else None,
        no_target_shot_share=float(np.mean(np.isnan(dist))) if s else None, hit_by_dist=hit_by_dist,
        fights=len(fs), fight_kill_share=float(np.mean([x['kill'] for x in fs])) if fs else None,
        fight_death_share=float(np.mean([x['died'] for x in fs])) if fs else None,
        fight_no_shot_share=float(np.mean([x['first_shot'] is None for x in fs])) if fs else None,
        steps_to_first_shot=float(np.median(first)) if first else None,
        fights_low_health_share=float(np.mean([x['health'] < 40 for x in fs])) if fs else None,
        low_health_combat_share=float(np.mean([r['low_health_combat'] for r in res])),
        close_bot_share=float(np.mean([r['close_no_shot'] for r in res])))


def parse_set(items):
    out = {}
    for kv in items:
        k, v = kv.split('=')
        out[k] = type(getattr(DM, k))(v)
    return out


if __name__ == '__main__':  # workers re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument('--matches', type=int, default=20)
    ap.add_argument('--seeds', choices=['val', 'fresh'], default='val')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--out', default='reports/d3_teacher_diag')
    a = ap.parse_args()
    cfg = replace(DM, **parse_set(a.set))
    seeds = {'val': VAL_SEEDS, 'fresh': FRESH_SEEDS}[a.seeds][:a.matches]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as ex:
        matches = list(ex.map(run_one, seeds, [cfg] * len(seeds)))
    res = [analyze(m) for m in matches]
    s = summary(res)
    with open(a.out + '.json', 'w') as fh:
        json.dump(dict(summary=s, per_match=[{k: v for k, v in r.items() if k not in ('shots', 'fights')} for r in res]),
                  fh, indent=1)
    print(json.dumps(s, indent=1))
