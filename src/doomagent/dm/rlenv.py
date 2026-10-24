"""E6, the game side of PPO (no torch here: one process per match stream, dm.ppo drives them).

A Runner plays the actions it is given. Every observation it returns carries, besides what the agent sees (the
newest frame, the vector, first), what only training reads: the teacher's label of this state (DMTeacher in the
shadow, for the kickstarting loss), the targets of the auxiliary and spatial heads in the recorder's format, the
critic's privileged vector, and the reward parts of the step that led to it. Labels and positions are read for the
reward, the targets and the critic only; the policy gets frames and the vector.
"""
import math
from collections import deque

import numpy as np

from doomagent.dm.auxtargets import DEPTH_SHAPE, POS_CELL, POS_NX, POS_NY, POS_ORIGIN, aux_step
from doomagent.dm.config import BOT, DM, MODES, TIC_RATE
from doomagent.dm.evaluate import STUCK_DIST, STUCK_STEPS, nearest_bot
from doomagent.dm.spatial import MAP_N, egomap
from doomagent.dm.teacher import DMTeacher, visible_bots
from doomagent.variants import BASE

# raw parts of a step's reward; dm.ppo weighs them (frags count 1, the rest is shaping)
PARTS = ('frag', 'death', 'damage', 'item', 'weapon', 'stuck')
N_PARTS = len(PARTS)
PRIV_NAMES = ('time', 'frags', 'deaths', 'lead', 'health', 'armor', 'ready', 'bots_seen', 'seen_dist', 'bot_dist',
              'bots_near', 'x', 'y', 'cos', 'sin', 'stuck') + tuple('mode_' + m for m in MODES)
PRIV_DIM = len(PRIV_NAMES)
FAR = 2048.0   # map units: distances are divided by this and capped at 1
NEAR = 512.0   # bots closer than this count as near


def match_steps(cfg=DM):
    return int(round(cfg.timelimit * 60 * TIC_RATE / cfg.frame_skip))


def step_parts(p, q, new_life, frag):
    # raw reward parts of one step from the privileged data before (p) and after (q) it: frags (a suicide is -1),
    # a death, damage dealt, an item picked up (the evaluator's rule: health, ammo of the same weapon or armor
    # rose), a weapon not owned before; 'stuck' is set by the Runner when the next observation arrives
    died = float(q.deaths > p.deaths)
    damage = max(float(q.damage - p.damage), 0.0)
    item = weapon = 0.0
    if not new_life:  # a respawn refills health and resets the weapons: not a pickup
        same = q.weapon == p.weapon
        item = float(q.health > p.health or (same and q.ammo > p.ammo) or q.armor > p.armor)
        weapon = float(sum(o > 0 for o in q.owned[2:]) > sum(o > 0 for o in p.owned[2:]))
    return np.array([frag, died, damage, item, weapon, 0.0], np.float32)


def stuck_now(hist):
    # hist: (x, y, bot or life start) of the last STUCK_STEPS + 1 observations. The step is stuck when the
    # player moved less than STUCK_DIST since the oldest one with no bot and no respawn in the last STUCK_STEPS:
    # dm.evaluate.stuck_steps, one step at a time
    if len(hist) <= STUCK_STEPS or any(e for _, _, e in list(hist)[1:]):
        return False
    (x0, y0, _), (x1, y1, _) = hist[0], hist[-1]
    return math.hypot(x1 - x0, y1 - y0) < STUCK_DIST


def critic_vec(p, mode, step, stuck, cfg=DM):
    # what the critic knows beyond the agent's input, scaled to about [-1, 1]: time left, the score against the
    # best bot, bots seen and bots anywhere (behind walls too), the place and heading, the teacher's mode
    seen = visible_bots(p.labels, p.px, p.py, cfg.width, p.vis_px, cfg.min_vis_px)
    dists = [math.hypot(o.x - p.px, o.y - p.py) for o in p.objects if o.name == BOT]
    bots = [float(f) for f in p.player_frags[1:]]
    a = math.radians(p.angle)
    v = [step / match_steps(cfg), p.frags / 20.0, p.deaths / 10.0,
         float(np.clip((p.frags - max(bots, default=0.0)) / 10.0, -2.0, 2.0)), p.health / 100.0, p.armor / 200.0,
         float(p.attack_ready), len(seen) / 7.0, min([e.dist for e in seen] + [FAR]) / FAR, min(dists + [FAR]) / FAR,
         sum(d < NEAR for d in dists) / 7.0, (p.px - POS_ORIGIN[0]) / (POS_NX * POS_CELL),
         (p.py - POS_ORIGIN[1]) / (POS_NY * POS_CELL), math.cos(a), math.sin(a), float(stuck)]
    onehot = [float(m == mode) for m in MODES]
    return np.array(v + onehot, np.float32)


class Runner:
    # one stream of matches: a variant (the base map with base_share, else a train variant) and a seed drawn per
    # match from its own generator; start() begins a match, step(action) plays one step. Both return the record
    # of the observation the next action is chosen on; a step that ends the match returns the next match's first
    # record with done=True and the finished match's numbers in 'match'.
    def __init__(self, cfg, variants, seeds, rng_seed, base_share=0.5, env_factory=None, teacher=None):
        if env_factory is None:
            from doomagent.dm.env import DeathmatchEnv
            env_factory = DeathmatchEnv
        self.cfg, self.seeds, self.base_share, self.env_factory = cfg, list(seeds), base_share, env_factory
        self.base = [v for v in variants if v.name == BASE.name]
        self.other = [v for v in variants if v.name != BASE.name]
        self.rng = np.random.default_rng(rng_seed)
        self.teacher = DMTeacher(cfg) if teacher is None else teacher
        self.env, self.hist, self.stats, self.t = None, deque(maxlen=STUCK_STEPS + 1), None, 0

    def _pick(self):
        if self.base and (not self.other or self.rng.random() < self.base_share):
            v = self.base[0]
        else:
            v = self.other[int(self.rng.integers(len(self.other)))]
        return v, int(self.seeds[int(self.rng.integers(len(self.seeds)))])

    def start(self, parts=None, done=False, match=None):
        v, seed = self._pick()
        if self.env is not None:
            self.env.close()
        self.env = self.env_factory(self.cfg, v)
        obs = self.env.reset(seed)
        self.teacher.reset()
        self.hist.clear()
        self.t = 0
        self.stats = dict(variant=v.name, seed=seed, parts=np.zeros(N_PARTS, np.float64))
        return self._observe(obs, np.zeros(N_PARTS, np.float32) if parts is None else parts, done, match)

    def step(self, action):
        p = self.env.privileged()
        obs, r, done = self.env.step(tuple(int(x) for x in action))
        q = self.env.privileged()
        parts = step_parts(p, q, obs.first, r)
        self.t += 1
        if done:
            self.stats['parts'] += parts
            return self.start(parts, True, self._summary(q))
        rec = self._observe(obs, parts, False, None)  # sets the stuck part
        self.stats['parts'] += parts
        return rec

    def _summary(self, q):
        s, n = self.stats, max(self.t, 1)
        bots = [float(f) for f in q.player_frags[1:]]
        return dict(variant=s['variant'], seed=s['seed'], length=self.t, frags=float(q.frags),
                    deaths=float(q.deaths), best_bot=max(bots, default=0.0), stuck_share=float(s['parts'][-1] / n),
                    parts={k: float(x) for k, x in zip(PARTS, s['parts'])})

    def _observe(self, obs, parts, done, match):
        p, c, t = self.env.privileged(), self.cfg, self.teacher
        label = tuple(int(x) for x in t(obs, p))
        enemy_x, items, depth = aux_step(p, c)
        self.hist.append((p.px, p.py, obs.first or nearest_bot(p.labels, p.px, p.py, c.width) is not None))
        stuck = stuck_now(self.hist)
        parts[-1] = float(stuck)
        mode = getattr(t, 'mode', 'patrol')
        goal = getattr(t, 'goal', None)
        nav = getattr(t, 'nav', None)
        return dict(
            frame=np.ascontiguousarray(obs.frames[-1]), vec=obs.vec, first=bool(obs.first),
            label=np.array(label, np.int64), mode=MODES.index(mode),
            pose=np.array([p.px, p.py, p.angle], np.float32), bot_visible=enemy_x >= 0, enemy_x=int(enemy_x),
            items=np.asarray(items, bool), depth_ok=depth is not None,
            depth=np.zeros(DEPTH_SHAPE, np.uint8) if depth is None else np.asarray(depth, np.uint8),
            map_ok=nav is not None,
            egomap=egomap(nav, p.px, p.py, p.angle) if nav is not None else np.zeros((MAP_N, MAP_N), bool),
            goal=np.array(goal[:2] if goal is not None and mode != 'combat' else (np.nan, np.nan), np.float32),
            critic=critic_vec(p, mode, self.t, stuck, c), parts=parts, done=bool(done), match=match)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


def worker(conn, cfg, variants, seeds, rng_seed, base_share):
    # a spawn process: ('start', None) or ('step', action) -> a record; ('close', None) ends it
    import doomagent.env  # noqa: F401  the Windows timer fixes, before any game starts
    runner = Runner(cfg, variants, seeds, rng_seed, base_share)
    try:
        while True:
            cmd, arg = conn.recv()
            if cmd == 'start':
                conn.send(runner.start())
            elif cmd == 'step':
                conn.send(runner.step(arg))
            else:
                break
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        runner.close()
