import sys
from collections import Counter

import numpy as np

from doomagent.config import N_ACTIONS
from doomagent.env import DoomEnv
from doomagent.teacher import MONSTERS, NON_ENEMIES
from doomagent.variants import get_variant

variant = get_variant(sys.argv[1] if len(sys.argv) > 1 else 'base')
env, rng, names = DoomEnv(variant=variant), np.random.default_rng(0), Counter()
for seed in range(5000, 5005):
    env.reset(seed)
    done = False
    while not done:
        for lb in env.privileged().labels:
            names[lb.object_name] += 1
        _, _, done = env.step(int(rng.integers(0, N_ACTIONS)))
env.close()
print(names.most_common())
print('dead-prefixed:', sorted(n for n in names if n.startswith('Dead'))
      or 'none: check how corpses are labelled in this build')
unknown = sorted(n for n in names if n.removeprefix('Dead') not in MONSTERS | NON_ENEMIES)
if unknown:
    print('UNKNOWN names, add each to MONSTERS or NON_ENEMIES in teacher.py:', unknown)
    sys.exit(1)
