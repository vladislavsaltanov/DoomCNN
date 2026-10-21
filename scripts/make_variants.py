import argparse
import json
import os

import numpy as np
import vizdoom as vzd

from doomagent.wadgen import (cover_sprites, map_textures, read_wad, sprite_lumps,
                              texture_pwad_lumps, write_wad)

# defend_the_center has two enemy classes; both are redrawn in every spr and mix variant.
# MarineChainsawVzd inherits ZDoom's MarineChainsaw, which is drawn with the PLAY sprite.
TARGETS = ('PLAY', 'SARG')  # chainsaw marine, demon
# donors never include the targets; BOSS is left out because it is a recolor of BOS2 (train)
DONORS = {'train': ['TROO', 'SPOS', 'POSS', 'BOS2', 'HEAD'], 'val': ['CPOS', 'SKEL'],
          'test': ['VILE', 'SSWV', 'BSPI']}
N_TEX = {'train': 16, 'val': 4, 'test': 4}
N_MIX = {'train': 8, 'val': 2, 'test': 4}

ap = argparse.ArgumentParser()
ap.add_argument('--doom2', help='path to doom2.wad, used as a test-only IWAD variant')
ap.add_argument('--out', default='variants')
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()

vz_dir = os.path.dirname(vzd.__file__)
walls, flats = map_textures(read_wad(os.path.join(vzd.scenarios_path, 'defend_the_center.wad')))
print('map wall textures:', walls)
print('map flats:', flats)
sprites = sprite_lumps(read_wad(os.path.join(vz_dir, 'freedoom2.wad')))
for s in TARGETS + tuple(d for ds in DONORS.values() for d in ds):
    assert any(n.startswith(s) for n in sprites), f'no {s} sprites in freedoom2.wad'
rng = np.random.default_rng(a.seed)
os.makedirs(a.out, exist_ok=True)


def save(name, split, kind, lumps=None, iwad=''):
    pwads = []
    if lumps is not None:
        path = f'{a.out}/{name}.wad'  # forward slash: the json is shared across OSes
        write_wad(path, lumps)
        pwads = [path]
    with open(os.path.join(a.out, name + '.json'), 'w') as fh:
        json.dump(dict(name=name, split=split, kind=kind, iwad=iwad, pwads=pwads), fh, indent=1)


def donor_pair(split, i):
    # marine and demon get different donors of the split
    d = DONORS[split]
    return d[i % len(d)], d[(i + 1) % len(d)]


def sprite_block(pair):
    lumps = [l for donor, target in zip(pair, TARGETS) for l in cover_sprites(sprites, donor, target)]
    return [('S_START', b'')] + lumps + [('S_END', b'')]


for split in ('train', 'val', 'test'):
    for i in range(N_TEX[split]):
        save(f'tex_{split}_{i:02d}', split, 'tex', texture_pwad_lumps(walls, flats, rng))
    for i in range(len(DONORS[split])):
        pair = donor_pair(split, i)
        save(f'spr_{split}_{pair[0].lower()}_{pair[1].lower()}', split, 'spr', sprite_block(pair))
    for i in range(N_MIX[split]):
        save(f'mix_{split}_{i:02d}', split, 'mix',
             texture_pwad_lumps(walls, flats, rng) + sprite_block(donor_pair(split, i)))
fd1 = os.path.join(vz_dir, 'freedoom1.wad')
if os.path.exists(fd1):
    save('iwad_freedoom1', 'val', 'iwad', iwad=fd1)
if a.doom2:
    save('iwad_doom2', 'test', 'iwad', iwad=a.doom2)
print('written to', a.out)
