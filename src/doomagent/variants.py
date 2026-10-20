import glob
import json
import os
from dataclasses import dataclass

import numpy as np

from doomagent.config import (DAGGER_SEEDS, TEST_OOD_EPISODES, TEST_SEEDS, VAL_OOD_EPISODES,
                              VAL_SEEDS)

VARIANTS_DIR = 'variants'


@dataclass(frozen=True)
class Variant:
    name: str = 'base'
    iwad: str = ''        # '' means the ViZDoom default IWAD (freedoom2.wad)
    pwads: tuple = ()
    split: str = 'train'  # train | val | test
    kind: str = 'base'    # base | tex | spr | mix | iwad

    def apply(self, game):
        if self.iwad:
            game.set_doom_game_path(os.path.abspath(self.iwad))
        for p in self.pwads:
            path = os.path.abspath(p)
            assert ' ' not in path, f'ViZDoom splits game args on spaces: {path}'
            game.add_game_args('-file ' + path)


BASE = Variant()


def load_variants(split=None, root=VARIANTS_DIR):
    out = []
    for p in sorted(glob.glob(os.path.join(root, '*.json'))):
        with open(p) as fh:
            d = json.load(fh)
        if d.get('disabled'):
            continue
        v = Variant(d['name'], d.get('iwad', ''), tuple(d.get('pwads', ())), d['split'],
                    d.get('kind', ''))
        if split is None or v.split == split:
            out.append(v)
    return out


def get_variant(name, root=VARIANTS_DIR):
    if name == 'base':
        return BASE
    for v in load_variants(root=root):
        if v.name == name:
            return v
    raise KeyError(name)


def spread_seeds(variants, seeds, per):
    # one disjoint chunk of seeds per variant
    assert len(variants) * per <= len(seeds), 'not enough seeds'
    return [(v, tuple(seeds[i * per:(i + 1) * per])) for i, v in enumerate(variants)]


def val_plan(root=VARIANTS_DIR):
    return [(BASE, VAL_SEEDS)] + [(v, VAL_SEEDS[:VAL_OOD_EPISODES])
                                  for v in load_variants('val', root)]


def test_plans(root=VARIANTS_DIR):
    return {'id': [(BASE, TEST_SEEDS)],
            'ood': [(v, TEST_SEEDS[:TEST_OOD_EPISODES]) for v in load_variants('test', root)]}


def dagger_plan(round_idx, per_round=100, root=VARIANTS_DIR):
    # rounds start at 2 (R0 and R1 are teacher recordings); seeds never repeat between rounds
    variants = [BASE] + load_variants('train', root)
    seeds = DAGGER_SEEDS[(round_idx - 2) * per_round:(round_idx - 1) * per_round]
    assert len(seeds) == per_round, 'DAGGER_SEEDS exhausted'
    pick = np.random.default_rng(round_idx).integers(0, len(variants), len(seeds))
    groups = {}
    for s, i in zip(seeds, pick):
        groups.setdefault(int(i), []).append(s)
    return [(variants[i], tuple(ss)) for i, ss in sorted(groups.items())]
