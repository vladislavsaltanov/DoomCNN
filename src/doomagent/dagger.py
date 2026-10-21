import argparse

import numpy as np
import torch

from doomagent.agent import StudentAgent, StudentChooser
from doomagent.config import Cfg
from doomagent.record import record_dataset
from doomagent.variants import dagger_plan


def student_executor(agent, beta, rng):
    # the student plays from pixels; with probability beta the teacher's label is executed
    def f(obs, priv, label):
        if beta > 0 and rng.random() < beta:
            return label
        return agent.act(obs)
    return f


class StudentExecutor:
    # picklable beta = 0 executor for parallel rounds: the student acts, the label is only recorded
    def __init__(self, model_path, cfg: Cfg = Cfg()):
        self.chooser = StudentChooser(torch.load(model_path, map_location='cpu'), cfg)

    def __call__(self, obs, priv, label):
        return self.chooser(obs, priv)


def run_round(model_path, out_dir, plan, beta=0.0, seed=0, cfg: Cfg = Cfg(), workers=1):
    if workers > 1:
        # beta > 0 draws from one generator across episodes; split up, the data would change
        assert beta == 0, 'parallel rounds need beta = 0'
        record_dataset(out_dir, plan, StudentExecutor(model_path, cfg), cfg, workers=workers)
        return
    agent = StudentAgent(model_path, cfg)
    record_dataset(out_dir, plan, student_executor(agent, beta, np.random.default_rng(seed)), cfg)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--round', type=int, required=True, help='2 for the first DAgger round')
    ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--per-round', type=int, default=100)
    ap.add_argument('--workers', type=int, default=1, help='processes; needs --beta 0')
    a = ap.parse_args()
    run_round(a.model, a.out, dagger_plan(a.round, a.per_round), a.beta, seed=a.round,
              workers=a.workers)
