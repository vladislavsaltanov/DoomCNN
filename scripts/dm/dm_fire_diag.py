"""Where the teacher's shots go: per shot of a hitscan weapon, how much of the target is on screen, whether
a straight 3D line from the gun to the target's body passes every opening on the way, and whether the
shot hit (HITCOUNT rose this step). Also how often a bot label sits behind a wall.

    python scripts/dm/dm_fire_diag.py [--seeds val|fresh] [--n 10] [--out reports/dm/fire_diag] [--set key=value ...]
"""
import argparse
import json
import math
import sys

import numpy as np

from doomagent.dm.config import BOT, DM, FRESH_SEEDS, PLAYER_HEIGHT, VAL_SEEDS
from doomagent.dm.env import DeathmatchEnv
from doomagent.dm.nav import shot_lines_from_sectors
from doomagent.dm.teacher import DMTeacher
from doomagent.dm.weapons import WEAPONS

SHOOT_Z = PLAYER_HEIGHT / 2 + 8.0  # a hitscan leaves the player this high above its feet
VIS_BINS = (0, 50, 200, 800, 3200, 10 ** 9)  # visible pixels of the target at 640x480


def los_3d(lines, px, py, pz, tx, ty, tz):
    # can a straight line from the gun reach some height of the target's body through every opening
    segs, lo, hi = lines
    x1, y1, x2, y2 = (segs[:, i] for i in range(4))
    dx, dy, ex, ey = tx - px, ty - py, x2 - x1, y2 - y1
    den = dx * ey - dy * ex
    with np.errstate(divide='ignore', invalid='ignore'):
        t = ((x1 - px) * ey - (y1 - py) * ex) / den
        u = ((x1 - px) * dy - (y1 - py) * dx) / den
    cross = (np.abs(den) > 1e-9) & (t > 1e-6) & (t < 1 - 1e-6) & (u >= 0) & (u <= 1)
    sz = pz + SHOOT_Z
    a, b = tz, tz + PLAYER_HEIGHT  # heights of the target's body the line may end at
    for ti, l, h in zip(t[cross], lo[cross], hi[cross]):
        if not np.isfinite(l) or h <= l:
            return False
        a, b = max(a, sz + (l - sz) / ti), min(b, sz + (h - sz) / ti)
        if a > b:
            return False
    return True


def bucket(v):
    for i in range(len(VIS_BINS) - 1):
        if VIS_BINS[i] <= v < VIS_BINS[i + 1]:
            return f'{VIS_BINS[i]}-{VIS_BINS[i + 1] if VIS_BINS[i + 1] < 10 ** 9 else ""}'


def run(seeds, cfg=DM):
    shots, labels_seen, steps, fires = [], dict(n=0, blocked=0, blocked_small=0), 0, 0
    teacher = DMTeacher(cfg)
    for seed in seeds:
        env = DeathmatchEnv(cfg)
        obs, done = env.reset(seed), False
        teacher.reset()
        lines = shot_lines_from_sectors(env.sectors)
        while not done:
            p = env.privileged()
            st = env.game.get_state()
            a = teacher(obs, p)
            if st is not None:
                z = {int(o.id): o.position_z for o in st.objects}
                me = min(st.objects, key=lambda o: math.hypot(o.position_x - p.px, o.position_y - p.py))
                pz = me.position_z
                buf = st.labels_buffer
                for lb in st.labels:
                    if lb.object_name != BOT or int(lb.object_id) == int(me.id):
                        continue
                    vis = int((buf == lb.value).sum())
                    ok = los_3d(lines, p.px, p.py, pz, lb.object_position_x, lb.object_position_y,
                                z.get(int(lb.object_id), pz))
                    labels_seen['n'] += 1
                    labels_seen['blocked'] += not ok
                    labels_seen['blocked_small'] += (not ok) and vis < 200
                w = WEAPONS.get(p.weapon)
                t = teacher.target
                if a[2] and t is not None and w is not None and w.speed == 0.0 and p.weapon != 1:
                    lb = next((x for x in st.labels if int(x.object_id) == t.id), None)
                    vis = int((buf == lb.value).sum()) if lb is not None else 0
                    ok = lb is not None and los_3d(lines, p.px, p.py, pz, lb.object_position_x,
                                                   lb.object_position_y, z.get(t.id, pz))
                    hits0 = p.hits
                    obs, _, done = env.step(a)
                    shots.append(dict(vis=vis, los=bool(ok), hit=bool(env.privileged().hits > hits0),
                                      dist=float(t.dist), weapon=w.name, seed=seed))
                    steps += 1
                    fires += 1
                    continue
            fires += int(a[2])
            obs, _, done = env.step(a)
            steps += 1
        env.close()
        print(f'seed {seed}: {len(shots)} hitscan shots so far', flush=True)
    return shots, labels_seen, steps, fires


def summary(shots, labels_seen, steps, fires):
    def rate(rows):
        return dict(n=len(rows), hit=float(np.mean([r['hit'] for r in rows])) if rows else None)
    out = dict(steps=steps, fire_share=fires / max(steps, 1), hitscan=rate(shots),
               by_los={str(k): rate([r for r in shots if r['los'] == k]) for k in (True, False)},
               by_vis={bucket(v): rate([r for r in shots if bucket(r['vis']) == bucket(v)]) for v in VIS_BINS[:-1]},
               by_weapon={n: rate([r for r in shots if r['weapon'] == n]) for n in sorted({r['weapon'] for r in shots})},
               bot_labels=dict(labels_seen, blocked_share=labels_seen['blocked'] / max(labels_seen['n'], 1)))
    return out


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', choices=['val', 'fresh'], default='val')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--out', default='reports/dm/fire_diag')
    ap.add_argument('--set', nargs='*', default=[])
    a = ap.parse_args()
    seeds = list(VAL_SEEDS[:a.n]) if a.seeds == 'val' else list(FRESH_SEEDS[:a.n])
    from dm_teacher_diag import with_overrides
    shots, labels_seen, steps, fires = run(seeds, with_overrides(DM, a.set))
    s = summary(shots, labels_seen, steps, fires)
    with open(a.out + '.json', 'w') as fh:
        json.dump(dict(summary=s, shots=shots), fh, indent=1)
    print(json.dumps(s, indent=1))
