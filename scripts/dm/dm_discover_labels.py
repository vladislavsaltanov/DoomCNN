import argparse
import sys
from collections import Counter

import numpy as np

from doomagent.dm.config import BARREL, BOT, DM, ITEM_KIND, N_ACTIONS, OTHER_NAMES, VAL_SEEDS
from doomagent.dm.env import DeathmatchEnv

ap = argparse.ArgumentParser(description='label names a random agent sees on cig MAP01')
ap.add_argument('--matches', type=int, default=5)
a = ap.parse_args()
env, rng, names = DeathmatchEnv(DM), np.random.default_rng(0), Counter()
for seed in VAL_SEEDS[:a.matches]:
    env.reset(seed)
    done = False
    while not done:
        for lb in env.privileged().labels:
            names[lb.object_name] += 1
        _, _, done = env.step(int(rng.integers(0, N_ACTIONS)))
env.close()
print(names.most_common())
unknown = sorted(set(names) - ({BOT, BARREL} | set(ITEM_KIND) | OTHER_NAMES))
if unknown:
    print('UNKNOWN names, classify each in doomagent/dm/config.py:', unknown)
    sys.exit(1)
print('all label names are classified')
