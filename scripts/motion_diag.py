"""Does the network use motion? Play with the real frame stack and with the last frame repeated
four times (no motion left in the input); a model that ignores motion loses nothing.

    python scripts/motion_diag.py runs/r4.pt runs/ppo_ft_flip/best_model.zip
"""
import sys

import numpy as np
import torch

from doomagent.agent import StudentChooser
from doomagent.evaluate import evaluate_plan, summarize
from doomagent.model import StudentCNN
from doomagent.variants import BASE, load_variants

SEEDS = tuple(range(6000, 6020))  # fresh seeds, as in scripts/s6_compare.py


class StillChooser(StudentChooser):
    def __call__(self, obs, priv):
        return super().__call__(np.repeat(np.asarray(obs)[-1:], len(obs), axis=0), priv)


def weights(path):
    if path.endswith('.zip'):
        from stable_baselines3 import PPO
        from doomagent.ppo import policy_state
        return policy_state(PPO.load(path, device='cpu'), StudentCNN().state_dict())
    return torch.load(path, map_location='cpu')


if __name__ == '__main__':
    plan = [(BASE, SEEDS)] + [(v, SEEDS) for v in load_variants('val')]
    for path in sys.argv[1:]:
        sd = weights(path)
        for name, chooser in (('4 frames', StudentChooser(sd)), ('last frame x4', StillChooser(sd))):
            res = evaluate_plan(plan, chooser, workers=4)
            base = summarize([r for r in res if r['variant'] == 'base'])
            ood = summarize([r for r in res if r['variant'] != 'base'])
            print(f"{path} | {name}: base {base['reward']['mean']:.2f} "
                  f"[{base['reward']['lo']:.2f}, {base['reward']['hi']:.2f}], "
                  f"ood {ood['reward']['mean']:.2f} [{ood['reward']['lo']:.2f}, "
                  f"{ood['reward']['hi']:.2f}], kills/shot {ood['kills_per_shot']:.3f}, "
                  f"blind {ood['blind_share']:.2f}, flips/100 {ood['flips_per_100_blind']:.1f}",
                  flush=True)
