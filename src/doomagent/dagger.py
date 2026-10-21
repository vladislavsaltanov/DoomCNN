import argparse

import numpy as np

from doomagent.agent import StudentAgent
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


def run_round(model_path, out_dir, plan, beta=0.0, seed=0, cfg: Cfg = Cfg()):
    agent = StudentAgent(model_path, cfg)
    record_dataset(out_dir, plan, student_executor(agent, beta, np.random.default_rng(seed)), cfg)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--round', type=int, required=True, help='2 for the first DAgger round')
    ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--per-round', type=int, default=100)
    a = ap.parse_args()
    run_round(a.model, a.out, dagger_plan(a.round, a.per_round), a.beta, seed=a.round)
