"""Why a student plays worse than its teacher: the student plays val matches while the teacher labels every step
in the shadow (as DAgger will). Agreement per head overall and in the teacher's combat steps, fire confusion,
shots, hits, time with a bot on screen, deaths.

    python scripts/dm/dm_student_diag.py runs/dm/dm2_m0_r0.pt --matches 6 --workers 2 --out runs/dm/student_diag
"""
import argparse
import json

import numpy as np

from doomagent.dm.actions import HEADS, TURN_DEG
from doomagent.dm.config import DM, VAL_SEEDS
from doomagent.dm.teacher import visible_bots


class Shadow:
    # plays the student, asks the teacher what it would do on the same step
    def __init__(self, state, memory):
        from doomagent.dm.agent import DMChooser
        self.student = DMChooser(state, memory, DM)
        self.teacher, self.rows = None, []

    def reset(self):
        from doomagent.dm.teacher import DMTeacher
        self.teacher = self.teacher or DMTeacher(DM)
        self.teacher.reset()
        self.student.reset()
        self.rows = []

    def __call__(self, obs, priv):
        t = self.teacher(obs, priv)
        s = self.student(obs, priv)
        bot = bool(visible_bots(priv.labels, priv.px, priv.py, DM.width, priv.vis_px, DM.min_vis_px))
        any_bot = bool(visible_bots(priv.labels, priv.px, priv.py, DM.width, priv.vis_px, 1.0))
        self.rows.append((tuple(t), tuple(s), self.teacher.mode, bot, any_bot))
        return s


def run_one(path, seed):
    from doomagent.dm.env import DeathmatchEnv
    from doomagent.dm.evaluate import run_match
    from doomagent.dm.model import load_model
    m = load_model(path, DM)
    ch = Shadow(m.state_dict(), m.memory)
    env = DeathmatchEnv(DM)
    try:
        r = run_match(env, ch, seed)
    finally:
        env.close()
    return dict(r, rows=ch.rows)


def analyze(matches):
    rows = [x for m in matches for x in m['rows']]
    t = np.array([r[0] for r in rows])
    s = np.array([r[1] for r in rows])
    combat = np.array([r[2] == 'combat' for r in rows])
    bot = np.array([r[3] for r in rows])
    any_bot = np.array([r[4] for r in rows])
    fire = s[:, 2].astype(bool)
    deg = np.array(TURN_DEG)
    out = dict(steps=len(rows),
               agree={h: float((t[:, k] == s[:, k]).mean()) for k, h in enumerate(HEADS)},
               agree_combat={h: float((t[combat, k] == s[combat, k]).mean()) for k, h in enumerate(HEADS)},
               combat_share=float(combat.mean()), bot_share=float(bot.mean()),
               teacher_fire=float(t[:, 2].mean()), student_fire=float(s[:, 2].mean()),
               fire_recall=float((s[:, 2] & t[:, 2]).sum() / max(t[:, 2].sum(), 1)),
               fire_precision=float((s[:, 2] & t[:, 2]).sum() / max(s[:, 2].sum(), 1)),
               turn_abs_err_combat=float(np.abs(deg[t[combat, 1]] - deg[s[combat, 1]]).mean()),
               turn_same_sign_combat=float((np.sign(deg[t[combat, 1]]) == np.sign(deg[s[combat, 1]])).mean()),
               student_fire_no_bot=float((fire & ~bot).sum() / max(fire.sum(), 1)),
               student_fire_tiny_bot=float((fire & ~bot & any_bot).sum() / max(fire.sum(), 1)),
               student_fire_nothing=float((fire & ~any_bot).sum() / max(fire.sum(), 1)),
               teacher_wants_next_weapon=float(t[:, 3].mean()), student_next_weapon=float(s[:, 3].mean()),
               weapon_share={k: float(np.mean([m['weapon_share'][k] for m in matches])) for k in range(1, 8)})
    for k in ('frags', 'deaths', 'shots', 'hits_per_shot', 'enemy_share', 'stuck_share', 'items', 'path'):
        out[k] = float(np.mean([m[k] for m in matches]))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--matches', type=int, default=6)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out', default='runs/dm/student_diag')
    a = ap.parse_args()
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    seeds = VAL_SEEDS[:a.matches]
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as ex:
        matches = list(ex.map(run_one, [a.model] * len(seeds), seeds))
    out = analyze(matches)
    with open(a.out + '.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
