"""Graphics variants of cig.wad MAP02 (DM.map) into variants_dm/: 16 train, 4 val, 6 test.

    python scripts/dm/dm_make_variants.py [--out variants_dm] [--seed 0]
"""
import argparse
import glob
import os

from doomagent.dm.config import DM, VARIANTS_DIR
from doomagent.dm.variants import make_variants, source_lumps

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=VARIANTS_DIR)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
for f in glob.glob(os.path.join(a.out, '*.json')) + glob.glob(os.path.join(a.out, '*.wad')):
    os.remove(f)  # the old variants (MAP01) go: a stale json would be loaded with the new ones
walls, flats, _ = source_lumps()
print(f'{DM.map.upper()}: {len(walls)} wall textures, {len(flats)} flats')
names = make_variants(a.out, a.seed)
for split in ('train', 'val', 'test'):
    print(split, sum(n.split('_')[1] == split for n in names), [n for n in names if n.split('_')[1] == split])
print(len(names), 'variants written to', a.out)
