"""Where on MAP02 the players spend their time: visit density per 64-unit cell and the share of walkable cells
ever visited, for the teacher (its recorded matches) and for models (positions saved by dm_neurons.py).

    python scripts/dm/dm_coverage.py --teacher data/dm/r0 --models runs/dm/dagger_sp_r2.acts.npz \
        runs/dm/e6/ppo_kl_03000k.acts.npz --names DAgger PPO --out reports/dm/e6_ppo/e6_coverage.png
"""
import argparse
import glob

import numpy as np

CELL = 64.0


def positions_teacher(folder, n):
    xy = [np.load(f)['pose'][:, :2] for f in sorted(glob.glob(folder + '/ep_*.npz'))[:n]]
    return np.concatenate(xy), len(xy)


def positions_model(path):
    z = np.load(path)
    return np.stack([z['px'], z['py']], 1), int(z['match'].max()) + 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', required=True)
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--matches', type=int, default=10, help='teacher matches (the models have what they played)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    from doomagent.dm.config import DM, VAL_SEEDS
    from doomagent.dm.env import DeathmatchEnv
    from doomagent.dm.nav import nav_for, walls_from_sectors
    from doomagent.dm.reports import INK, MUTED, SURFACE, plt
    env = DeathmatchEnv(DM)
    env.reset(VAL_SEEDS[0])
    sectors = env.sectors
    env.close()
    walls, nav = walls_from_sectors(sectors), nav_for(sectors)
    x0, y0 = nav.x0, nav.y0
    nx, ny = int(np.ceil(nav.nx * nav.cell / CELL)), int(np.ceil(nav.ny * nav.cell / CELL))
    # walkable 64-unit cells: any walkable nav cell inside
    k = int(CELL // nav.cell)
    walk = np.zeros((ny, nx), bool)
    for i in range(ny):
        for j in range(nx):
            walk[i, j] = nav.walk[i * k:(i + 1) * k, j * k:(j + 1) * k].any()
    runs = [('teacher',) + positions_teacher(a.teacher, a.matches)] + \
        [(n,) + positions_model(p) for n, p in zip(a.names, a.models)]
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 5.0), facecolor=SURFACE)
    for ax, (name, xy, matches) in zip(np.atleast_1d(axes), runs):
        j = np.clip(((xy[:, 0] - x0) // CELL).astype(int), 0, nx - 1)
        i = np.clip(((xy[:, 1] - y0) // CELL).astype(int), 0, ny - 1)
        grid = np.zeros((ny, nx))
        np.add.at(grid, (i, j), 1.0)
        seen = (grid > 0) & walk
        share = seen.sum() / walk.sum()
        dens = np.where(grid > 0, np.log1p(grid / matches), np.nan)
        ax.imshow(np.ma.masked_invalid(dens), cmap='Blues', origin='lower', interpolation='nearest',
                  extent=(x0, x0 + nx * CELL, y0, y0 + ny * CELL))
        for x1, y1, x2, y2 in walls:
            ax.plot((x1, x2), (y1, y2), color='#898781', linewidth=0.6)
        ax.set_title(f'{name}: {matches} matches, {share:.0%} of the walkable map visited', color=MUTED, fontsize=9,
                     loc='left')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        for s in ax.spines.values():
            s.set_visible(False)
        print(f'{name}: {share:.1%} of walkable cells visited in {matches} matches')
    fig.suptitle('Where the players spend their time on MAP02 (darker = more steps; white = never there)', color=INK,
                 fontsize=11, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(a.out, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    print(a.out)
