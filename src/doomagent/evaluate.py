from collections import defaultdict

import numpy as np

from doomagent.config import SHOOT, Cfg

IDLE_STEPS = 100


def pixel_only(act):
    # adapts an act(obs) function to the chooser(obs, priv) interface, dropping priv
    return lambda obs, priv: act(obs)


def run_episode(env, chooser, seed):
    if hasattr(chooser, 'reset'):
        chooser.reset()
    obs = env.reset(seed)
    ammo = env.privileged().ammo
    total, kills, shots, pressed, steps, first_shot, done = 0.0, 0.0, 0, 0, 0, None, False
    while not done:
        action = int(chooser(obs, env.privileged()))
        pressed += action == SHOOT
        obs, r, done = env.step(action)
        if not done:  # the terminal state carries no game variables
            new_ammo = env.privileged().ammo
            if new_ammo < ammo:
                shots += int(round(ammo - new_ammo))
                if first_shot is None:
                    first_shot = steps
            ammo = new_ammo
        total += r
        kills += max(r, 0.0)
        steps += 1
    return dict(reward=total, kills=kills, shots=shots, pressed=pressed, length=steps,
                first_shot=first_shot)


def _run_chunk(variant, seeds, chooser, cfg, env_factory):
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


def evaluate_plan(plan, chooser, cfg: Cfg = Cfg(), env_factory=None, workers=1):
    # workers > 1: episodes run in separate processes (chooser and env_factory must pickle);
    # each episode depends only on its seed, so the results equal the serial ones, in plan order
    if workers <= 1:
        return [r for v, seeds in plan for r in _run_chunk(v, seeds, chooser, cfg, env_factory)]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    total = sum(len(s) for _, s in plan)
    size = max(1, -(-total // (2 * workers)))  # about two chunks per worker
    chunks = [(v, tuple(seeds[i:i + size])) for v, seeds in plan for i in range(0, len(seeds), size)]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as ex:
        futures = [ex.submit(_run_chunk, v, s, chooser, cfg, env_factory) for v, s in chunks]
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
    return out
