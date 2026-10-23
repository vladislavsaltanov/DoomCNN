import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

from doomagent.dm import config as C
from doomagent.dm.actions import HEAD_SIZES
from doomagent.dm.config import BOT, DM, MODES
from doomagent.variants import BASE, load_variants, spread_seeds


class MatchNoise:
    # executor: the teacher's label; with probability p the move or the turn head (half and half) gets another
    # random value; fire and weapon are never changed. Reseeded per match, so parallel recording equals serial.
    def __init__(self, p=DM.noise_p):
        self.p, self.rng = p, np.random.default_rng(0)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, priv, label):
        a = [int(x) for x in label]
        if self.rng.random() < self.p:
            head = int(self.rng.integers(2))
            v = int(self.rng.integers(HEAD_SIZES[head] - 1))
            a[head] = v + (v >= a[head])  # any value but the label's
        return tuple(a)


def record_match(env, seed, executor, teacher, prefix, cfg=DM):
    # one whole match: the teacher labels every step, the executor decides what is played
    teacher.reset()
    if hasattr(executor, 'reset'):
        executor.reset(seed)
    obs, done = env.reset(seed), False
    rows = {k: [] for k in ('actions', 'executed', 'vec', 'first', 'rewards', 'modes', 'pose',
                            'stats', 'counters', 'bot_visible', 'weapon')}
    frames = []
    while not done:
        p = env.privileged()
        label = tuple(int(x) for x in teacher(obs, p))
        action = tuple(int(x) for x in executor(obs, p, label))
        frames.append(obs.frames[-1].copy())
        rows['actions'].append(label)
        rows['executed'].append(action)
        rows['vec'].append(obs.vec)
        rows['first'].append(obs.first)
        rows['modes'].append(MODES.index(teacher.mode))
        rows['pose'].append((p.px, p.py, p.angle))
        rows['stats'].append((p.health, p.armor, p.ammo, p.items))
        rows['counters'].append((p.frags, p.deaths, p.hits))
        rows['bot_visible'].append(any(lb.object_name == BOT for lb in p.labels))
        rows['weapon'].append(p.weapon)
        obs, r, done = env.step(action)
        rows['rewards'].append(r)
    np.save(prefix + '.frames.npy', np.stack(frames).astype(np.uint8))
    dtypes = dict(actions=np.int64, executed=np.int64, vec=np.float32, first=bool, rewards=np.float32,
                  modes=np.int8, pose=np.float32, stats=np.float32, counters=np.float32,
                  bot_visible=bool, weapon=np.int8)
    np.savez(prefix + '.npz', **{k: np.array(v, dtype=dtypes[k]) for k, v in rows.items()},
             seed=seed, variant=env.variant.name, scenario='deathmatch')
    return len(frames)


def episode_prefix(out_dir, variant, seed):
    return os.path.join(out_dir, f'ep_{variant.name}_{seed}')


def write_meta(out_dir):
    steps, n, per = 0, 0, Counter()
    for f in sorted(glob.glob(os.path.join(out_dir, 'ep_*.npz'))):
        z = np.load(f)
        steps += len(z['actions'])
        n += 1
        per[str(z['variant'])] += 1
    with open(os.path.join(out_dir, 'meta.json'), 'w') as fh:
        json.dump(dict(matches=n, steps=steps, variants=dict(per)), fh, indent=1)


def record_chunk(out_dir, variant, seeds, executor, cfg):
    from doomagent.dm.env import DeathmatchEnv
    from doomagent.dm.teacher import DMTeacher
    teacher, env = DMTeacher(cfg), DeathmatchEnv(cfg, variant)
    try:
        for s in seeds:
            record_match(env, s, executor, teacher, episode_prefix(out_dir, variant, s), cfg)
    finally:
        env.close()


def record_dataset(out_dir, plan, executor, cfg=DM, resume=True, workers=1):
    # plan: [(variant, seeds)]; finished matches are skipped when resuming
    os.makedirs(out_dir, exist_ok=True)
    chunks = []
    for variant, seeds in plan:
        todo = [s for s in seeds
                if not (resume and os.path.exists(episode_prefix(out_dir, variant, s) + '.npz'))]
        if todo:
            chunks.append((variant, todo))
    if workers <= 1:
        for variant, todo in chunks:
            record_chunk(out_dir, variant, todo, executor, cfg)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        total = sum(len(t) for _, t in chunks)
        size = max(1, -(-total // (2 * workers)))
        parts = [(v, t[i:i + size]) for v, t in chunks for i in range(0, len(t), size)]
        with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as ex:
            for f in [ex.submit(record_chunk, out_dir, v, t, executor, cfg) for v, t in parts]:
                f.result()
    write_meta(out_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--set', choices=['base', 'variants', 'pilot'], default='base')
    ap.add_argument('--matches', type=int, default=150)
    ap.add_argument('--per-variant', type=int, default=6)
    ap.add_argument('--noise', type=float, default=DM.noise_p)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    if a.set == 'base':
        plan = [(BASE, C.TRAIN_SEEDS[:a.matches])]
    elif a.set == 'pilot':
        plan = [(BASE, C.PILOT_SEEDS)]
    else:
        plan = spread_seeds(load_variants('train', C.VARIANTS_DIR), C.VARIANT_SEEDS, a.per_variant)
    record_dataset(a.out, plan, MatchNoise(a.noise), DM, workers=a.workers)
    print(open(os.path.join(a.out, 'meta.json')).read())
