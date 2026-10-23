"""Neuron maps of a deathmatch network from its own matches (a shadow teacher names the mode only).

    python scripts/dm_neurons.py --model runs/dm_m1_pilot.pt --plan val10 --tag m1 --prefix artifacts/d2_

Writes {prefix}place_{tag}.png, {prefix}select_{tag}.png, {prefix}states_{tag}.png and the
activations next to the model (runs/<name>.acts.npz).
"""
import argparse

from doomagent.dm.config import DM, TEST_SEEDS, VAL_SEEDS, val_plan
from doomagent.dm.env import DeathmatchEnv
from doomagent.dm.neurons import CONDITIONS, place_maps, record_activations, selectivity, state_map
from doomagent.dm.reports import place_figure, selectivity_figure, states_figure
from doomagent.variants import BASE

if __name__ == '__main__':  # evaluation workers re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--plan', choices=['val10', 'val', 'test'], default='val10')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--prefix', default='artifacts/d2_')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    plan = {'val10': lambda: [(BASE, VAL_SEEDS[:10])], 'val': val_plan,
            'test': lambda: [(BASE, TEST_SEEDS)]}[a.plan]()
    acts = record_activations(a.model, plan, DM, workers=a.workers)
    print(f'{len(acts["act"])} steps from {int(acts["match"].max()) + 1} matches')
    env = DeathmatchEnv(DM)
    env.reset(VAL_SEEDS[0])
    sectors = env.sectors
    env.close()
    place_figure(place_maps(acts), sectors, f'{a.prefix}place_{a.tag}.png')
    best, strength, _ = selectivity(acts)
    selectivity_figure(best, strength, CONDITIONS, f'{a.prefix}select_{a.tag}.png')
    xy, mode = state_map(acts)
    states_figure(xy, mode, f'{a.prefix}states_{a.tag}.png')
    for k, name in enumerate(CONDITIONS):
        sel = (best == k) & (strength >= 0.1)
        print(f'{name}: {int(sel.sum())} units, strongest |r| {strength[best == k].max(initial=0):.2f}')
    print('written', f'{a.prefix}{{place,select,states}}_{a.tag}.png')
