"""How fast the teacher gets its aim on a bot: steps from taking a target to the first shot, and the steps it
does not fire although the turn it chose ends within the fire tolerance (fire is decided on the error before
the turn).

    python scripts/dm/dm_aim_diag.py [--n 6] [--set key=value ...]
"""
import argparse
import json
import sys

import numpy as np

from doomagent.dm.actions import TURN_DEG
from doomagent.dm.aim import half_width_deg, target_angle
from doomagent.dm.config import DM, VAL_SEEDS
from doomagent.dm.env import DeathmatchEnv
from doomagent.dm.teacher import DMTeacher

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.path.insert(0, 'scripts/dm')
    from dm_teacher_diag import with_overrides
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--set', nargs='*', default=[])
    a = ap.parse_args()
    cfg = with_overrides(DM, a.set)
    to_shot, err, after, fired_after, combat, fires, frags = [], [], [], 0, 0, 0, []
    for seed in VAL_SEEDS[:a.n]:
        env, t = DeathmatchEnv(cfg), DMTeacher(cfg)
        obs, done, cur, since = env.reset(seed), False, None, 0
        while not done:
            p = env.privileged()
            act = t(obs, p)
            if t.mode == 'combat' and t.target is not None:
                combat += 1
                rel = target_angle(p.px, p.py, p.angle, *t.aim)
                tol = cfg.aim_tol * half_width_deg(t.target.dist)
                err.append(abs(rel))
                if t.target.id != cur:
                    cur, since = t.target.id, 0
                if act[2]:
                    fires += 1
                    if since >= 0:
                        to_shot.append(since)
                        since = -10 ** 9
                elif t.why == 'not_aimed':
                    after.append(abs(rel - TURN_DEG[act[1]]) <= tol)
                since += 1
            else:
                cur = None
            obs, _, done = env.step(act)
        frags.append(env.privileged().frags)
        env.close()
    to_shot = np.array(to_shot)
    out = dict(matches=a.n, frags=float(np.mean(frags)), combat_steps=combat, fire_share_in_combat=fires / max(combat, 1),
               steps_to_first_shot=dict(median=float(np.median(to_shot)), mean=float(to_shot.mean()),
                                        share_0=float(np.mean(to_shot == 0)), share_ge3=float(np.mean(to_shot >= 3)),
                                        n=int(len(to_shot))),
               aim_error_deg=dict(median=float(np.median(err)), p90=float(np.percentile(err, 90))),
               not_aimed_but_on_target_after_turn=float(np.mean(after)) if after else None)
    print(json.dumps(out, indent=1))
