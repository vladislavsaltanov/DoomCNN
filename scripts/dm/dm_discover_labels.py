import argparse
import sys
from collections import Counter

from doomagent.dm.config import (ARMOR_ITEMS, AMMO_ITEMS, BARREL, BOT, DM, HEALTH_ITEMS, OTHER_NAMES, PROJECTILES,
                                 VAL_SEEDS, WEAPON_ITEMS)
from doomagent.dm.env import DeathmatchEnv
from doomagent.dm.evaluate import RandomChooser

ap = argparse.ArgumentParser(description='label and object names a random player meets on cig MAP02')
ap.add_argument('--matches', type=int, default=5)
a = ap.parse_args()
env, chooser, labels, objects = DeathmatchEnv(DM), RandomChooser(), Counter(), Counter()
for seed in VAL_SEEDS[:a.matches]:
    chooser.reset()
    obs, done = env.reset(seed), False
    while not done:
        p = env.privileged()
        labels.update(lb.object_name for lb in p.labels)
        objects.update(o.name for o in p.objects)
        obs, _, done = env.step(chooser(obs, p))
env.close()
print('labels:', labels.most_common())
print('objects:', objects.most_common())
known = {BOT, BARREL} | set(WEAPON_ITEMS) | set(AMMO_ITEMS) | HEALTH_ITEMS | ARMOR_ITEMS | set(PROJECTILES) | OTHER_NAMES
unknown = sorted((set(labels) | set(objects)) - known)
if unknown:
    print('UNKNOWN names, classify each in doomagent/dm/config.py:', unknown)
    sys.exit(1)
print('all names are classified')
