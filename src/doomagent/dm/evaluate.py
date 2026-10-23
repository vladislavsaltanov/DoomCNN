import math

import numpy as np

from doomagent.dm.actions import HEAD_SIZES, N_WEAPON_SLOTS
from doomagent.dm.config import BOT, DM
from doomagent.evaluate import bootstrap_ci

STUCK_STEPS = 17   # 2 s of agent steps
STUCK_DIST = 32.0  # map units: moving less than this over STUCK_STEPS with no bot around is stuck
TRACE_KEYS = ('px', 'py', 'angle', 'action', 'first', 'enemy', 'enemy_x', 'enemy_wx', 'enemy_wy',
              'health', 'ammo', 'frags', 'items', 'weapon')
SUMMARY_KEYS = ('frags', 'deaths', 'kills', 'suicides', 'kd', 'hits_per_shot', 'items', 'path',
                'stuck_share', 'enemy_share', 'bot_median', 'bot_best')
FIRE_P, WEAPON_P = 0.3, 0.05  # random player: fire and next-weapon press probabilities


class RandomChooser:
    # move and turn uniform, fire with FIRE_P, next weapon with WEAPON_P; the generator is seeded from the
    # match's first frame, so a match repeats in any worker process
    def reset(self):
        self.rng, self.actions = None, []

    def __call__(self, obs, priv):
        if self.rng is None:
            self.rng = np.random.default_rng(int(obs.frames[-1].sum()))
        g = self.rng
        a = (int(g.integers(0, HEAD_SIZES[0])), int(g.integers(0, HEAD_SIZES[1])), int(g.random() < FIRE_P),
             int(g.random() < WEAPON_P))
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


def run_match(env, chooser, seed, trace=False, tap=None):
    # chooser(obs, priv) -> action; game data is read here for the metrics only; tap (dm.livetap.LiveTap)
    # shows the match on the dashboard
    if hasattr(chooser, 'reset'):
        chooser.reset()
    w = getattr(env, 'cfg', DM).width
    obs = env.reset(seed)
    rows = {k: [] for k in TRACE_KEYS}
    pos, enemy, first = [], [], []
    shots, suicides, pickups, path, length, done = 0, 0, 0, 0.0, 0, False
    weapon_steps = np.zeros(N_WEAPON_SLOTS + 1)
    while not done:
        p = env.privileged()
        bot = nearest_bot(p.labels, p.px, p.py, w)
        action = tuple(int(x) for x in chooser(obs, p))
        if tap is not None:
            tap(env, p, getattr(chooser, 'teacher', chooser if hasattr(chooser, 'mode') else None), action, action,
                dict(seed=seed, step=length, variant=env.variant.name))
        weapon_steps[min(max(int(p.weapon), 0), N_WEAPON_SLOTS)] += 1
        pos.append((p.px, p.py))
        enemy.append(bot is not None)
        first.append(bool(obs.first))
        if trace:
            nan = float('nan')
            for k, v in zip(TRACE_KEYS, (p.px, p.py, p.angle, action, obs.first, bot is not None,
                                         bot[1] if bot else nan, bot[2] if bot else nan,
                                         bot[3] if bot else nan, p.health, p.ammo, p.frags, p.items,
                                         p.weapon)):
                rows[k].append(v)
        obs, r, done = env.step(action)
        q = env.privileged()
        suicides += r < 0  # only a suicide lowers the frag count
        if not obs.first:  # a respawn resets the weapons and moves the player: not a shot, no path
            same = q.weapon == p.weapon  # a switch changes the ammo shown, it is not a shot or a pickup
            if same and q.ammo < p.ammo:
                shots += int(round(p.ammo - q.ammo))
            path += math.hypot(q.px - p.px, q.py - p.py)
            # ITEMCOUNT skips medikits and ammo: a pickup is a rise in health, ammo in hand or armor
            pickups += q.health > p.health or (same and q.ammo > p.ammo) or q.armor > p.armor
        length += 1
    end = env.privileged()
    bots = [float(f) for f in end.player_frags[1:]]
    stuck = stuck_steps(pos, enemy, first)
    out = dict(reward=float(end.frags), frags=float(end.frags), deaths=float(end.deaths),
               suicides=int(suicides), kills=float(end.frags) + suicides,
               kd=float(end.frags) / max(float(end.deaths), 1.0), shots=shots,
               hits=float(end.hits), hits_per_shot=float(end.hits) / max(shots, 1),
               items=float(pickups), path=path, stuck_steps=stuck,
               stuck_share=stuck / max(length, 1), enemy_steps=int(sum(enemy)),
               enemy_share=sum(enemy) / max(length, 1), length=length, bot_frags=bots,
               bot_median=float(np.median(bots)) if bots else None, bot_best=max(bots) if bots else None,
               rank=1 + sum(b > float(end.frags) for b in bots),
               weapon_share={k: float(weapon_steps[k] / max(length, 1)) for k in range(1, N_WEAPON_SLOTS + 1)})
    if tap is not None:
        tap.match_done(seed, frags=out['frags'], deaths=out['deaths'], best_bot=out['bot_best'] or 0.0,
                       first=out['rank'] == 1)
    if trace:
        out['trace'] = {k: np.asarray(v) for k, v in rows.items()}
        if hasattr(chooser, 'take_trace'):
            out['chooser'] = chooser.take_trace()
    return out


def run_chunk(variant, seeds, chooser, cfg, env_factory=None, trace=False, live=None):
    # live: a folder (dm.livetap.LIVE_DIR) where this worker shows its matches on the dashboard
    tap = None
    if live:
        from doomagent.dm.livetap import LiveTap
        tap = LiveTap(live, cfg=cfg)
    if env_factory is None:
        from doomagent.dm.env import DeathmatchEnv
        env_factory = DeathmatchEnv
    out = []
    env = env_factory(cfg, variant)
    try:
        for s in seeds:
            r = run_match(env, chooser, s, trace, tap)
            r.update(variant=variant.name, seed=int(s))
            out.append(r)
    finally:
        env.close()
    return out


def evaluate_plan(plan, chooser, cfg=DM, env_factory=None, workers=1, trace=False, live=None):
    # plan: [(variant, seeds)]; workers > 1 runs matches in spawn processes (chooser must pickle);
    # a match depends only on its seed, so the results equal the serial ones, in plan order
    if workers <= 1:
        return [r for v, seeds in plan for r in run_chunk(v, seeds, chooser, cfg, env_factory, trace, live)]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    total = sum(len(s) for _, s in plan)
    size = max(1, -(-total // (2 * workers)))  # about two chunks per worker
    chunks = [(v, tuple(seeds[i:i + size])) for v, seeds in plan for i in range(0, len(seeds), size)]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as ex:
        futures = [ex.submit(run_chunk, v, s, chooser, cfg, env_factory, trace, live) for v, s in chunks]
        return [r for f in futures for r in f.result()]


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
    out['first_share'] = float(np.mean([r['rank'] == 1 for r in results]))
    out['weapon_share'] = {k: float(np.mean([r['weapon_share'][k] for r in results]))
                           for k in range(1, N_WEAPON_SLOTS + 1)}
    return out
