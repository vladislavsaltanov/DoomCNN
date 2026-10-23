"""Where the cheater teacher loses on MAP02: deaths, shots by weapon, fights, the weapon in hand.

    python scripts/dm/dm_teacher_diag.py [--seeds val|fresh] [--matches 20] [--workers 4] [--set key=value ...]
        [--out runs/dm/diag_<name>]

Per death: a suicide (frags went down), a bot on screen in the 2 s before or not, the mode and health 2 s
before. Per shot (ammo in hand went down, same weapon): did the hit counter rise within 1.5 s, by weapon.
Per step with a bot on screen: the teacher's reason for not firing (teacher.why). Per life: steps until a
weapon of rank >= 3 was in hand. Plus the place among the 8 players and the best bot. The version 1.1 script
for MAP01 is at the tag deathmatch-v1.
"""
import argparse
import json
from dataclasses import fields, replace

import numpy as np

from doomagent.dm.config import DM, FRESH_SEEDS, MODES, VAL_SEEDS
from doomagent.dm.teacher import visible_bots
from doomagent.dm.weapons import WEAPONS

WINDOW = 17      # 2 s of agent steps
HIT_WINDOW = 13  # 1.5 s
WHY = ('shoot', 'switching', 'not_aimed', 'unsafe', 'no_ammo', 'reloading', 'no_combat')


_TAP = None  # one LiveTap per worker process: the dashboard shows what it plays


def run_one(seed, cfg, live=False):
    global _TAP
    from doomagent.dm.env import DeathmatchEnv
    from doomagent.dm.teacher import DMTeacher
    if live and _TAP is None:
        from doomagent.dm.livetap import LiveTap
        _TAP = LiveTap(cfg=cfg)
    env, teacher = DeathmatchEnv(cfg), DMTeacher(cfg)
    obs, done, rows = env.reset(seed), False, []
    while not done:
        p = env.privileged()
        a = teacher(obs, p)
        if live:
            _TAP(env, p, teacher, a, a, dict(seed=seed, step=len(rows), variant=env.variant.name))
        rows.append(dict(why=teacher.why, mode=MODES.index(teacher.mode), health=p.health, ammo=p.weapon_ammo,
                         weapon=p.weapon, frags=p.frags, deaths=p.deaths, hits=p.hits, first=bool(obs.first),
                         bot=bool(visible_bots(p.labels, p.px, p.py, cfg.width)), fire=int(a[2])))
        obs, _, done = env.step(a)
    end = env.privileged()
    env.close()
    if live:
        bots = [float(f) for f in end.player_frags[1:]]
        _TAP.match_done(seed, frags=float(end.frags), deaths=float(end.deaths), best_bot=max(bots, default=0.0),
                        first=not any(b > float(end.frags) for b in bots))  # rank 1, as summary() counts it
    return dict(seed=seed, rows=rows, player_frags=[float(f) for f in end.player_frags])


def analyze(m):
    rows = m['rows']
    col = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    T = len(rows)
    deaths, shots, to_gear, life, got = [], [], [], 0, None
    for t in range(1, T):
        if col['deaths'][t] > col['deaths'][t - 1]:  # before the new life starts: the death step is also its first
            w0 = max(life, t - WINDOW)
            deaths.append(dict(suicide=bool(col['frags'][t] < col['frags'][t - 1]), bot_seen=bool(col['bot'][w0:t].any()),
                               mode=MODES[int(col['mode'][t - 1])], health_before=float(col['health'][w0]),
                               weapon=int(col['weapon'][t - 1])))
        if col['first'][t]:
            to_gear.append(got if got is not None else t - life)
            life, got = t, None
        if got is None and WEAPONS.get(int(col['weapon'][t]), WEAPONS[2]).rank >= 3:
            got = t - life
        if col['ammo'][t] < col['ammo'][t - 1] and col['weapon'][t] == col['weapon'][t - 1] and not col['first'][t]:
            shots.append(dict(weapon=int(col['weapon'][t]),
                              hit=bool(col['hits'][min(t - 1 + HIT_WINDOW, T - 1)] > col['hits'][t - 1])))
    why = {w: int(((col['why'] == w) & col['bot']).sum()) for w in WHY}
    pf = m['player_frags']
    return dict(seed=m['seed'], frags=pf[0], deaths=deaths, shots=shots, to_gear=to_gear, why=why,
                rank=1 + sum(f > pf[0] for f in pf[1:]), best_bot=max(pf[1:]), median_bot=float(np.median(pf[1:])),
                weapon_share={k: float((col['weapon'] == k).mean()) for k in WEAPONS})


def summary(res):
    d = [x for r in res for x in r['deaths']]
    s = [x for r in res for x in r['shots']]
    n = max(len(d), 1)
    why = {w: sum(r['why'][w] for r in res) for w in WHY}
    nw = max(sum(why.values()), 1)
    by_w = {WEAPONS[k].name: (float(np.mean([x['hit'] for x in s if x['weapon'] == k])),
                              sum(x['weapon'] == k for x in s)) for k in WEAPONS if any(x['weapon'] == k for x in s)}
    gear = [g for r in res for g in r['to_gear']]
    return dict(
        matches=len(res), frags=float(np.mean([r['frags'] for r in res])),
        deaths=len(d) / max(len(res), 1), best_bot=float(np.mean([r['best_bot'] for r in res])),
        median_bot=float(np.mean([r['median_bot'] for r in res])),
        first_share=float(np.mean([r['rank'] == 1 for r in res])), rank_mean=float(np.mean([r['rank'] for r in res])),
        kd_total=sum(r['frags'] for r in res) / max(len(d), 1),
        death_cause={'suicide': sum(x['suicide'] for x in d) / n,
                     'bot seen': sum(x['bot_seen'] and not x['suicide'] for x in d) / n,
                     'unseen': sum(not x['bot_seen'] and not x['suicide'] for x in d) / n},
        death_mode={m: sum(x['mode'] == m for x in d) / n for m in MODES},
        suicide_weapon={WEAPONS[k].name: sum(x['suicide'] and x['weapon'] == k for x in d) for k in WEAPONS},
        death_health_before=float(np.mean([x['health_before'] for x in d])) if d else None,
        shots_per_match=len(s) / max(len(res), 1),
        hit_share_by_weapon={k: v[0] for k, v in by_w.items()}, shots_by_weapon={k: v[1] for k, v in by_w.items()},
        bot_on_screen_by_reason={w: v / nw for w, v in why.items()},
        steps_to_gear_median=float(np.median(gear)) if gear else None,
        weapon_share={WEAPONS[k].name: float(np.mean([r['weapon_share'][k] for r in res])) for k in WEAPONS})


def with_overrides(cfg, items):
    types = {f.name: type(getattr(cfg, f.name)) for f in fields(cfg)}
    return replace(cfg, **{k: (v == 'True') if types[k] is bool else types[k](v)
                           for k, v in (x.split('=', 1) for x in items or [])})


if __name__ == '__main__':  # workers re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', choices=['val', 'fresh'], default='val')
    ap.add_argument('--matches', type=int, default=20)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--out', default='runs/dm/diag_teacher')
    ap.add_argument('--no-live', action='store_true', help='no frames for the dashboard')
    a = ap.parse_args()
    cfg = with_overrides(DM, a.set)
    seeds = {'val': VAL_SEEDS, 'fresh': FRESH_SEEDS}[a.seeds][:a.matches]
    live = not a.no_live
    if live:
        from doomagent.dm.livetap import write_plan
        write_plan(f'проверка учителя · {" ".join(a.set) or "по умолчанию"} · {a.seeds}', '', len(seeds), kind='eval')
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as ex:
        matches = list(ex.map(run_one, seeds, [cfg] * len(seeds), [live] * len(seeds)))
    res = [analyze(m) for m in matches]
    s = summary(res)
    with open(a.out + '.json', 'w') as fh:
        json.dump(dict(set=a.set, seeds=a.seeds, summary=s,
                       per_match=[{k: v for k, v in r.items() if k not in ('shots', 'deaths')} for r in res]), fh, indent=1)
    print(json.dumps(s, indent=1))
