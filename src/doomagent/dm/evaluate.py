import math

import numpy as np

from doomagent.dm.config import BOT, DM, N_ACTIONS
from doomagent.evaluate import bootstrap_ci

STUCK_STEPS = 17   # 2 s of agent steps
STUCK_DIST = 32.0  # map units: moving less than this over STUCK_STEPS with no bot around is stuck
TRACE_KEYS = ('px', 'py', 'angle', 'action', 'first', 'enemy', 'enemy_x', 'enemy_wx', 'enemy_wy',
              'health', 'ammo', 'frags', 'items')
SUMMARY_KEYS = ('frags', 'deaths', 'kills', 'suicides', 'kd', 'hits_per_shot', 'items', 'path',
                'stuck_share', 'enemy_share', 'bot_median')


class RandomChooser:
    # uniform over the ten actions; the generator is seeded from the match's first frame, so a match
    # repeats in any worker process
    def reset(self):
        self.rng, self.actions = None, []

    def __call__(self, obs, priv):
        if self.rng is None:
            self.rng = np.random.default_rng(int(obs.frames[-1].sum()))
        a = int(self.rng.integers(0, N_ACTIONS))
        self.actions.append(a)
        return a


def nearest_bot(labels, px, py, screen_w):
    # (distance, screen offset in [-1, 1], world x, world y) of the nearest visible bot, or None
    best = None
    for lb in labels:
        if lb.object_name != BOT:
            continue
        d = math.hypot(lb.object_position_x - px, lb.object_position_y - py)
        if best is None or d < best[0]:
            best = (d, 2.0 * (lb.x + lb.width / 2.0) / screen_w - 1.0, lb.object_position_x,
                    lb.object_position_y)
    return best


def stuck_steps(pos, enemy, first):
    # steps that moved less than STUCK_DIST over the last STUCK_STEPS with no bot and no respawn
    n = 0
    for t in range(STUCK_STEPS, len(pos)):
        window = range(t - STUCK_STEPS + 1, t + 1)
        if any(enemy[i] or first[i] for i in window):
            continue
        (x0, y0), (x1, y1) = pos[t - STUCK_STEPS], pos[t]
        n += math.hypot(x1 - x0, y1 - y0) < STUCK_DIST
    return n


def run_match(env, chooser, seed, trace=False):
    # chooser(obs, priv) -> action; game data is read here for the metrics only
    if hasattr(chooser, 'reset'):
        chooser.reset()
    w = getattr(env, 'cfg', DM).width
    obs = env.reset(seed)
    rows = {k: [] for k in TRACE_KEYS}
    pos, enemy, first = [], [], []
    shots, suicides, path, length, done = 0, 0, 0.0, 0, False
    while not done:
        p = env.privileged()
        bot = nearest_bot(p.labels, p.px, p.py, w)
        action = int(chooser(obs, p))
        pos.append((p.px, p.py))
        enemy.append(bot is not None)
        first.append(bool(obs.first))
        if trace:
            nan = float('nan')
            for k, v in zip(TRACE_KEYS, (p.px, p.py, p.angle, action, obs.first, bot is not None,
                                         bot[1] if bot else nan, bot[2] if bot else nan,
                                         bot[3] if bot else nan, p.health, p.ammo, p.frags, p.items)):
                rows[k].append(v)
        obs, r, done = env.step(action)
        q = env.privileged()
        suicides += r < 0  # only a suicide lowers the frag count
        if not obs.first:  # a respawn refills the rockets and moves the player: not a shot, no path
            if q.ammo < p.ammo:
                shots += int(round(p.ammo - q.ammo))
            path += math.hypot(q.px - p.px, q.py - p.py)
        length += 1
    end = env.privileged()
    bots = [float(f) for f in end.player_frags[1:]]
    stuck = stuck_steps(pos, enemy, first)
    out = dict(reward=float(end.frags), frags=float(end.frags), deaths=float(end.deaths),
               suicides=int(suicides), kills=float(end.frags) + suicides,
               kd=float(end.frags) / max(float(end.deaths), 1.0), shots=shots,
               hits=float(end.hits), hits_per_shot=float(end.hits) / max(shots, 1),
               items=float(end.items), path=path, stuck_steps=stuck,
               stuck_share=stuck / max(length, 1), enemy_steps=int(sum(enemy)),
               enemy_share=sum(enemy) / max(length, 1), length=length, bot_frags=bots,
               bot_median=float(np.median(bots)) if bots else None)
    if trace:
        out['trace'] = {k: np.asarray(v) for k, v in rows.items()}
        if hasattr(chooser, 'take_trace'):
            out['chooser'] = chooser.take_trace()
    return out


def run_chunk(variant, seeds, chooser, cfg, env_factory=None, trace=False):
    if env_factory is None:
        from doomagent.dm.env import DeathmatchEnv
        env_factory = DeathmatchEnv
    out = []
    env = env_factory(cfg, variant)
    try:
        for s in seeds:
            r = run_match(env, chooser, s, trace)
            r.update(variant=variant.name, seed=int(s))
            out.append(r)
    finally:
        env.close()
    return out


def summarize(results):
    out = {}
    for key in SUMMARY_KEYS:
        values = [r[key] for r in results if r.get(key) is not None]
        if values:
            m, lo, hi = bootstrap_ci(values)
            out[key] = dict(mean=m, lo=lo, hi=hi)
    out['reward'] = out['frags']  # evaluate.val_score and the reports read 'reward'
    out['kd_total'] = sum(r['frags'] for r in results) / max(sum(r['deaths'] for r in results), 1.0)
    out['shots'] = float(np.mean([r['shots'] for r in results]))
    out['length'] = float(np.mean([r['length'] for r in results]))
    return out
