import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

from doomagent import config as C
from doomagent.config import FINE_LEFT, FINE_RIGHT, SHOOT, TURN_LEFT, TURN_RIGHT, Cfg
from doomagent.teacher import Teacher, enemies_from_labels, enemy_mask, nearest_aux
from doomagent.variants import BASE, load_variants, spread_seeds

TURNS = (TURN_LEFT, TURN_RIGHT, FINE_LEFT, FINE_RIGHT)


def noisy(p, rng):
    # executor: the teacher's label, sometimes replaced by a random turn; never injects a shot
    def f(obs, priv, label):
        if label != SHOOT and rng.random() < p:
            return int(TURNS[rng.integers(0, len(TURNS))])
        return label
    return f


def record_episode(env, seed, executor, teacher, prefix, cfg: Cfg = Cfg()):
    teacher.reset()
    obs = env.reset(seed)
    rows = {k: [] for k in ('actions', 'vis', 'off', 'rewards', 'ammo')}
    frames, masks, done = [], [], False
    while not done:
        priv = env.privileged()
        label = teacher(obs, priv)
        v, o = nearest_aux(enemies_from_labels(priv.labels, priv.px, priv.py, cfg.width))
        frames.append(obs[-1].copy())
        masks.append(np.packbits(enemy_mask(priv, cfg), axis=1))
        rows['actions'].append(label)
        rows['vis'].append(v)
        rows['off'].append(o)
        rows['ammo'].append(priv.ammo)
        obs, r, done = env.step(int(executor(obs, priv, label)))
        rows['rewards'].append(r)
    np.save(prefix + '.frames.npy', np.stack(frames).astype(np.uint8))
    np.save(prefix + '.masks.npy', np.stack(masks))
    ammo = np.array(rows['ammo'], dtype=np.float32)
    np.savez(prefix + '.npz', actions=np.array(rows['actions'], dtype=np.int64),
             vis=np.array(rows['vis'], dtype=np.int64),
             off=np.array(rows['off'], dtype=np.float32),
             rewards=np.array(rows['rewards'], dtype=np.float32),
             ammo=ammo, valid=ammo > 0, seed=seed, variant=env.variant.name)
    return len(rows['actions'])


def episode_prefix(out_dir, variant, seed):
    return os.path.join(out_dir, f'ep_{variant.name}_{seed}')


def write_meta(out_dir):
    steps, n, per = 0, 0, Counter()
    for f in glob.glob(os.path.join(out_dir, 'ep_*.npz')):
        z = np.load(f)
        steps += len(z['actions'])
        n += 1
        per[str(z['variant'])] += 1
    with open(os.path.join(out_dir, 'meta.json'), 'w') as fh:
        json.dump(dict(episodes=n, steps=steps, variants=dict(per)), fh, indent=1)


def record_dataset(out_dir, plan, executor, cfg: Cfg = Cfg(), resume=True):
    from doomagent.env import DoomEnv
    os.makedirs(out_dir, exist_ok=True)
    teacher = Teacher(cfg)
    for variant, seeds in plan:
        todo = [s for s in seeds
                if not (resume and os.path.exists(episode_prefix(out_dir, variant, s) + '.npz'))]
        if not todo:
            continue
        env = DoomEnv(cfg, variant)
        try:
            for s in todo:
                record_episode(env, s, executor, teacher, episode_prefix(out_dir, variant, s), cfg)
        finally:
            env.close()
    write_meta(out_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--set', choices=['base', 'base_extra', 'variants'], default='base')
    ap.add_argument('--per-variant', type=int, default=15)
    ap.add_argument('--noise', type=float, default=Cfg().noise_p)
    a = ap.parse_args()
    if a.set == 'base':
        plan = [(BASE, C.TRAIN_SEEDS)]
    elif a.set == 'base_extra':
        plan = [(BASE, C.TRAIN_SEEDS_EXTRA)]
    else:
        plan = spread_seeds(load_variants('train'), C.VARIANT_SEEDS, a.per_variant)
    record_dataset(a.out, plan, noisy(a.noise, np.random.default_rng(0)))
