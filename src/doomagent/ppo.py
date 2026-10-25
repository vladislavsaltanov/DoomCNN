import argparse
import json
import os
from functools import partial

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from doomagent.agent import StudentChooser
from doomagent.config import N_ACTIONS, RL_SEEDS, Cfg
from doomagent.env import DoomEnv
from doomagent.evaluate import SIDE, evaluate_plan, val_score
from doomagent.model import StudentCNN
from doomagent.teacher import is_enemy_name
from doomagent.variants import BASE, load_variants, val_plan


class FlipPenalty:
    # reward shaping (S7 decision): a left/right reversal of the turn while no enemy
    # is on screen costs `penalty`; seeing an enemy resets the direction
    def __init__(self, penalty):
        self.penalty = penalty
        self.reset()

    def reset(self):
        self.side = None

    def __call__(self, action, blind):
        if not blind:
            self.side = None
            return 0.0
        if action not in SIDE:
            return 0.0
        flip = self.side is not None and SIDE[action] != self.side
        self.side = SIDE[action]
        return -self.penalty if flip else 0.0


class DoomGymEnv(gym.Env):
    # pixels only: info is always empty, so PPO never sees privileged data; the labels are read
    # only to compute the shaping term of the reward
    def __init__(self, seeds, variants=(BASE,), switch_every=5, rng_seed=0, cfg: Cfg = Cfg(),
                 flip_penalty=0.0):
        super().__init__()
        self.flip = FlipPenalty(flip_penalty) if flip_penalty > 0 else None
        self.cfg, self.seeds, self.variants = cfg, list(seeds), list(variants)
        self.switch_every, self.episodes, self.env = switch_every, 0, None
        self.rng = np.random.default_rng(rng_seed)
        self.observation_space = gym.spaces.Box(0, 255, (cfg.stack, cfg.height, cfg.width),
                                                np.uint8)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

    def _maybe_switch(self):
        due = len(self.variants) > 1 and self.episodes % self.switch_every == 0
        if self.env is None or due:
            if self.env is not None:
                self.env.close()
            v = self.variants[int(self.rng.integers(len(self.variants)))]
            self.env = DoomEnv(self.cfg, v)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._maybe_switch()
        self.episodes += 1
        if self.flip is not None:
            self.flip.reset()
        return self.env.reset(int(self.rng.choice(self.seeds))), {}

    def step(self, action):
        shaping = 0.0
        if self.flip is not None:
            blind = not any(is_enemy_name(lb.object_name) for lb in self.env.privileged().labels)
            shaping = self.flip(int(action), blind)
        obs, r, done = self.env.step(int(action))
        return obs, r + shaping, done, False, {}

    def close(self):
        if self.env is not None:
            self.env.close()


def make_vec(seeds, variants, n_envs, seed=0, cfg: Cfg = Cfg(), flip_penalty=0.0):
    fns = [partial(DoomGymEnv, tuple(seeds), tuple(variants), 5, seed + i, cfg, flip_penalty)
           for i in range(n_envs)]
    return VecMonitor(SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns))


class StudentExtractor(BaseFeaturesExtractor):
    # same layers as StudentCNN; SB3 already scales uint8 images to [0, 1]
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=512)
        s = StudentCNN()
        self.body, self.fc = s.body, s.fc

    def forward(self, obs):
        return self.fc(self.body(obs).flatten(1))


def make_ppo(venv, student, lr, seed=0, ent_coef=0.01):
    kw = dict(features_extractor_class=StudentExtractor, net_arch=dict(pi=[], vf=[]),
              share_features_extractor=False)
    model = PPO('CnnPolicy', venv, learning_rate=lr, n_steps=512, batch_size=256, n_epochs=4,
                gamma=0.99, gae_lambda=0.95, clip_range=0.1, ent_coef=ent_coef,
                policy_kwargs=kw, seed=seed, verbose=1)
    if student is not None:
        # both extractors start from the student; the critic gets its own copy
        for fe in (model.policy.pi_features_extractor, model.policy.vf_features_extractor):
            fe.body.load_state_dict(student.body.state_dict())
            fe.fc.load_state_dict(student.fc.state_dict())
        model.policy.action_net.load_state_dict(student.pi.state_dict())
    return model


def policy_state(model, base_state):
    # the PPO actor as StudentCNN weights (body, fc, pi); the auxiliary heads come from
    # base_state and do not affect the action
    fe, out = model.policy.pi_features_extractor, dict(base_state)
    for prefix, module in (('body.', fe.body), ('fc.', fe.fc), ('pi.', model.policy.action_net)):
        for k, v in module.state_dict().items():
            out[prefix + k] = v.detach().cpu().clone()
    return out


def set_lr(model, lr):
    model.lr_schedule = lambda _progress: lr


def freeze_actor(model, frozen):
    params = (list(model.policy.pi_features_extractor.parameters())
              + list(model.policy.action_net.parameters()))
    for p in params:
        p.requires_grad_(not frozen)


class ValCallback(BaseCallback):
    # the same val_score as imitation; the first evaluation runs before any update,
    # so best_model.zip can never be worse than the policy PPO started from
    def __init__(self, plan, out_dir, eval_freq, cfg: Cfg = Cfg(), workers=1, base_state=None):
        super().__init__()
        self.plan, self.out_dir, self.eval_freq, self.cfg = plan, out_dir, eval_freq, cfg
        self.workers = workers
        self.base_state = base_state or StudentCNN(cfg).state_dict()
        self.best, self.last, self.log = -float('inf'), 0, []
        os.makedirs(out_dir, exist_ok=True)

    def _on_training_start(self):
        if not self.log:
            self._evaluate()

    def _on_step(self):
        if self.num_timesteps - self.last >= self.eval_freq:
            self._evaluate()
        return True

    def _evaluate(self):
        self.last = self.num_timesteps
        # the actor runs as a StudentCNN copy, so the episodes can run in worker processes
        chooser = StudentChooser(policy_state(self.model, self.base_state), self.cfg)
        score, per = val_score(evaluate_plan(self.plan, chooser, self.cfg, workers=self.workers))
        self.log.append(dict(timesteps=int(self.num_timesteps), val_score=score,
                             per_variant=per))
        print(self.log[-1], flush=True)
        with open(os.path.join(self.out_dir, 'val_log.json'), 'w') as fh:
            json.dump(self.log, fh, indent=1)
        if score > self.best:
            self.best = score
            self.model.save(os.path.join(self.out_dir, 'best_model'))


def _variants(variants):
    return [BASE] + load_variants('train') if variants is None else list(variants)


def finetune(student_path, out_dir, warmup_steps, steps, variants=None, n_envs=4, lr_warm=1e-4,
             lr=1e-5, seed=0, eval_freq=20_000, val=None, cfg: Cfg = Cfg(), flip_penalty=0.0,
             eval_workers=1):
    student = StudentCNN(cfg)
    student.load_state_dict(torch.load(student_path, map_location='cpu'))
    venv = make_vec(RL_SEEDS, _variants(variants), n_envs, seed, cfg, flip_penalty)
    model = make_ppo(venv, student, lr_warm, seed, ent_coef=0.001)
    cb = ValCallback(val_plan() if val is None else val, out_dir, eval_freq, cfg, eval_workers,
                     student.state_dict())
    freeze_actor(model, True)  # critic warm-up
    set_lr(model, lr_warm)
    model.learn(warmup_steps, callback=cb)
    freeze_actor(model, False)
    set_lr(model, lr)
    model.learn(steps, callback=cb, reset_num_timesteps=False)
    venv.close()
    return os.path.join(out_dir, 'best_model.zip')


def scratch(out_dir, steps, variants=None, n_envs=4, lr=2.5e-4, seed=0, eval_freq=20_000,
            val=None, cfg: Cfg = Cfg(), eval_workers=1):
    venv = make_vec(RL_SEEDS, _variants(variants), n_envs, seed, cfg)
    model = make_ppo(venv, None, lr, seed)
    model.learn(steps, callback=ValCallback(val_plan() if val is None else val, out_dir,
                                            eval_freq, cfg, eval_workers))
    venv.close()
    return os.path.join(out_dir, 'best_model.zip')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('finetune')
    f.add_argument('--student', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--warmup', type=int, default=20_000)
    f.add_argument('--steps', type=int, default=180_000)
    s = sub.add_parser('scratch')
    s.add_argument('--out', required=True)
    s.add_argument('--steps', type=int, required=True)
    f.add_argument('--flip-penalty', type=float, default=0.0,
                   help='reward shaping: cost of a left/right reversal with no enemy on screen')
    for p in (f, s):
        p.add_argument('--n-envs', type=int, default=4)
        p.add_argument('--variants', choices=['train', 'base'], default='train')
        p.add_argument('--eval-workers', type=int, default=1)
    a = ap.parse_args()
    variants = [BASE] if a.variants == 'base' else None
    if a.cmd == 'finetune':
        print(finetune(a.student, a.out, a.warmup, a.steps, variants, a.n_envs,
                       flip_penalty=a.flip_penalty, eval_workers=a.eval_workers))
    else:
        print(scratch(a.out, a.steps, variants, a.n_envs, eval_workers=a.eval_workers))
