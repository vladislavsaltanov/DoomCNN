from collections import defaultdict

import numpy as np

from doomagent.config import FINE_LEFT, FINE_RIGHT, SHOOT, TURN_LEFT, TURN_RIGHT, Cfg
from doomagent.teacher import is_enemy_name

IDLE_STEPS = 100
SIDE = {TURN_LEFT: -1, FINE_LEFT: -1, TURN_RIGHT: 1, FINE_RIGHT: 1}  # turn direction


def pixel_only(act):
    # adapts an act(obs) function to the chooser(obs, priv) interface, dropping priv
    return lambda obs, priv: act(obs)


def run_episode(env, chooser, seed):
    if hasattr(chooser, 'reset'):
        chooser.reset()
    obs = env.reset(seed)
    ammo = env.privileged().ammo
    total, kills, shots, pressed, steps, first_shot, done = 0.0, 0.0, 0, 0, 0, None, False
    # search: steps with no enemy on screen (labels, evaluator only), the longest such streak,
    # and left/right reversals while blind (spinning in place); None without labels
    ammo_out, blind, streak, longest, flips, side = None, 0, 0, 0, 0, None
    has_labels = hasattr(env.privileged(), 'labels')
    while not done:
        priv = env.privileged()
        action = int(chooser(obs, priv))
        pressed += action == SHOOT
        if has_labels:
            if any(is_enemy_name(lb.object_name) for lb in priv.labels):
                streak, side = 0, None
            else:
                blind += 1
                streak += 1
                longest = max(longest, streak)
                if action in SIDE:
                    flips += side is not None and SIDE[action] != side
                    side = SIDE[action]
        obs, r, done = env.step(action)
        if not done:  # the terminal state carries no game variables
            new_ammo = env.privileged().ammo
            if new_ammo < ammo:
                shots += int(round(ammo - new_ammo))
                if first_shot is None:
                    first_shot = steps
            ammo = new_ammo
            if ammo <= 0 and ammo_out is None:
                ammo_out = steps + 1
        total += r
        kills += max(r, 0.0)
        steps += 1
    return dict(reward=total, kills=kills, shots=shots, pressed=pressed, length=steps,
                first_shot=first_shot, ammo_out=ammo_out,
                blind_steps=blind if has_labels else None,
                longest_blind=longest if has_labels else None,
                blind_flips=flips if has_labels else None)


def _run_chunk(variant, seeds, chooser, cfg, env_factory, trace=False):
    if cfg.scenario == 'deathmatch':
        from doomagent.dm.evaluate import run_chunk
        return run_chunk(variant, seeds, chooser, cfg, env_factory, trace)
    if env_factory is None:
        from doomagent.env import DoomEnv
        env_factory = DoomEnv
    out = []
    env = env_factory(cfg, variant)
    try:
        for s in seeds:
            r = run_episode(env, chooser, s)
            r.update(variant=variant.name, seed=int(s))
            out.append(r)
    finally:
        env.close()
    return out


def evaluate_plan(plan, chooser, cfg: Cfg = Cfg(), env_factory=None, workers=1, trace=False):
    # workers > 1: episodes run in separate processes (chooser and env_factory must pickle);
    # each episode depends only on its seed, so the results equal the serial ones, in plan order.
    # trace: per-step arrays in each deathmatch result (defend_the_center ignores it)
    if workers <= 1:
        return [r for v, seeds in plan
                for r in _run_chunk(v, seeds, chooser, cfg, env_factory, trace)]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    total = sum(len(s) for _, s in plan)
    size = max(1, -(-total // (2 * workers)))  # about two chunks per worker
    chunks = [(v, tuple(seeds[i:i + size])) for v, seeds in plan for i in range(0, len(seeds), size)]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as ex:
        futures = [ex.submit(_run_chunk, v, s, chooser, cfg, env_factory, trace) for v, s in chunks]
        return [r for f in futures for r in f.result()]


def val_score(results):
    # mean over variants of the mean reward: every variant weighs the same
    groups = defaultdict(list)
    for r in results:
        groups[r['variant']].append(r['reward'])
    per = {k: float(np.mean(v)) for k, v in groups.items()}
    return float(np.mean(list(per.values()))), per


def bootstrap_ci(values, n_boot=2000, seed=0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(a, b):
    return bootstrap_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def summarize(results):
    out = {}
    for key in ('reward', 'kills', 'shots', 'length'):
        m, lo, hi = bootstrap_ci([r[key] for r in results])
        out[key] = dict(mean=m, lo=lo, hi=hi)
    shots = sum(r['shots'] for r in results)
    out['kills_per_shot'] = sum(r['kills'] for r in results) / max(shots, 1)
    out['pressed_per_ep'] = float(np.mean([r['pressed'] for r in results]))
    idle = [r['first_shot'] is None or r['first_shot'] > IDLE_STEPS for r in results]
    out['idle_rate'] = float(np.mean(idle))
    out_steps = [r['ammo_out'] for r in results if r.get('ammo_out') is not None]
    out['ammo_out_rate'] = len(out_steps) / max(len(results), 1)
    out['ammo_out_step'] = float(np.mean(out_steps)) if out_steps else None
    seen = [r for r in results if r.get('blind_steps') is not None]
    blind = sum(r['blind_steps'] for r in seen)
    out['blind_share'] = blind / max(sum(r['length'] for r in seen), 1) if seen else None
    out['longest_blind'] = float(np.mean([r['longest_blind'] for r in seen])) if seen else None
    out['flips_per_100_blind'] = (100.0 * sum(r['blind_flips'] for r in seen) / max(blind, 1)
                                  if seen else None)
    return out
