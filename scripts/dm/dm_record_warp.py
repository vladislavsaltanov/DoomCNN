"""The teacher's matches with half of the lives starting at a random walkable spot (record.Warper): data that shows the
whole map, the rooms the game never spawns in included. Seeds no earlier recording used: the base map on DAGGER_SEEDS
600-669, the train variants on VARIANT_SEEDS from 500 (150 matches by default).

    python scripts/dm/dm_record_warp.py --out data/dm/r2w
"""
import argparse
import json
import os
from dataclasses import replace

if __name__ == '__main__':  # recording workers re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', type=int, default=70)
    ap.add_argument('--per-variant', type=int, default=5)
    ap.add_argument('--warp', type=float, default=0.5)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()

    from doomagent.dm import config as C
    from doomagent.dm.config import DM
    from doomagent.dm.record import MatchNoise, record_dataset
    from doomagent.variants import BASE, load_variants, spread_seeds
    plan = [(BASE, C.DAGGER_SEEDS[600:600 + a.base])] + spread_seeds(load_variants('train', C.VARIANTS_DIR),
                                                                       C.VARIANT_SEEDS[500:], a.per_variant)
    record_dataset(a.out, plan, MatchNoise(DM.noise_p, DM.burst_p), replace(DM, depth=True, cheats=True),
                   workers=a.workers, live=False, warp_p=a.warp)
    with open(os.path.join(a.out, 'meta.json')) as fh:
        print(json.dumps(json.load(fh)))
