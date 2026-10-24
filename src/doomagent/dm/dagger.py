import argparse
from dataclasses import replace

import numpy as np
import torch

from doomagent.dm.agent import DMChooser
from doomagent.dm.config import DM, dagger_plan
from doomagent.dm.record import record_dataset


class DAggerExecutor:
    # the student plays: its network runs on every step (its memory follows the real trajectory), with
    # probability beta the teacher's labels are executed instead. The teacher labels every step and the recorder
    # writes the targets of all heads (the egomap, the goal, depth, pose) on the student's own states.
    # Reseeded per match, so parallel recording equals serial; picklable (DMChooser builds the net in the worker)
    def __init__(self, state, memory, beta, cfg=DM):
        self.chooser, self.beta = DMChooser(state, memory, cfg), beta
        self.rng, self.in_burst = np.random.default_rng(0), False

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.chooser.reset()

    def __call__(self, obs, priv, label):
        a = self.chooser(obs, priv)
        return tuple(int(x) for x in label) if self.beta > 0 and self.rng.random() < self.beta else a


def stop_rule(scores, teacher_frags, min_gain=0.05, max_rounds=3):
    # scores[0]: the model before DAgger, then one per round. A round's gain is its score minus the best before it;
    # stop after two rounds in a row gaining at most min_gain * teacher_frags, or after max_rounds rounds
    small = [s - max(scores[:i]) <= min_gain * teacher_frags for i, s in enumerate(scores) if i]
    stop = len(small) >= max_rounds or (len(small) >= 2 and small[-1] and small[-2])
    return stop, int(np.argmax(scores))


def run_round(model_path, out_dir, plan, beta, cfg=DM, workers=4, live=True):
    d = torch.load(model_path, map_location='cpu', weights_only=True)
    record_dataset(out_dir, plan, DAggerExecutor(d['state'], d['memory'], beta, cfg), replace(cfg, depth=True),
                   workers=workers, live=live)


if __name__ == '__main__':  # worker processes re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--round', type=int, required=True, help='1 for the first DAgger round')
    ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--per-round', type=int, default=40)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no-live', action='store_true')
    a = ap.parse_args()
    run_round(a.model, a.out, dagger_plan(a.round, a.per_round), a.beta, workers=a.workers, live=not a.no_live)
