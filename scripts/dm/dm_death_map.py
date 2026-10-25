"""Where players die and where the map itself hurts them: death spots and environmental damage (health lost with no
bot and no projectile in the labels for HAZARD_CALM steps: lava, slime, the player's own rocket splash) over a few
matches, drawn on the MAP02 outline.

    python scripts/dm/dm_death_map.py --who teacher runs/dm/e6/ppo_e6c_01001k.pt --names teacher E6c \
        --seeds 39000 39010 --out reports/dm/e7_test/e7_deaths.png
"""
import argparse
import json

import numpy as np

HAZARD_CALM = 4  # steps (0.46 s) with no bot and no projectile in the labels before a health loss


def play(who, seeds):
    from doomagent.dm.agent import DMAgent
    from doomagent.dm.config import BOT, DM, PROJECTILES
    from doomagent.dm.env import DeathmatchEnv
    from doomagent.dm.model import load_model
    from doomagent.dm.teacher import DMTeacher
    env = DeathmatchEnv(DM)
    agent = None if who == 'teacher' else DMAgent(load_model(who, DM))
    teacher = DMTeacher(DM) if agent is None else None
    deaths, hurt, sectors = [], [], None
    try:
        for seed in seeds:
            obs = env.reset(seed)
            sectors = env.sectors
            if agent is not None:
                agent.reset()
            else:
                teacher.reset()
            done, calm = False, 0
            while not done:
                p = env.privileged()
                danger = any(lb.object_name == BOT or lb.object_name in PROJECTILES for lb in p.labels)
                calm = 0 if danger else calm + 1
                action = agent.act(obs.frames, obs.vec, obs.first) if agent is not None else teacher(obs, p)
                obs, _, done = env.step(action)
                q = env.privileged()
                if q.deaths > p.deaths:
                    deaths.append((p.px, p.py, calm >= HAZARD_CALM))
                elif not obs.first and q.health < p.health and calm >= HAZARD_CALM:
                    hurt.append((p.px, p.py, p.health - q.health))
    finally:
        env.close()
    return dict(deaths=deaths, hurt=hurt), sectors


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--who', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--seeds', nargs=2, type=int, default=[39000, 39010])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from doomagent.dm.nav import walls_from_sectors
    from doomagent.dm.reports import INK, MUTED, SURFACE, plt
    seeds = list(range(*a.seeds))
    runs, sectors = [], None
    for who, name in zip(a.who, a.names):
        r, sectors = play(who, seeds)
        runs.append((name, r))
        hz = sum(h for _, _, h in r['hurt'])
        print(f"{name}: {len(r['deaths'])} deaths in {len(seeds)} matches, {sum(d[2] for d in r['deaths'])} of them "
              f"with no bot or projectile around; environmental damage {hz:.0f} HP", flush=True)
    walls = walls_from_sectors(sectors)
    fig, axes = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 6.2), facecolor=SURFACE)
    for ax, (name, r) in zip(np.atleast_1d(axes), runs):
        for x1, y1, x2, y2 in walls:
            ax.plot((x1, x2), (y1, y2), color='#898781', linewidth=0.6)
        if r['hurt']:
            h = np.array(r['hurt'])
            ax.scatter(h[:, 0], h[:, 1], s=4 + 2 * h[:, 2], color='#eb6834', alpha=0.35,
                       label='health lost with nobody around (lava, own rockets)')
        d = np.array([(x, y) for x, y, quiet in r['deaths'] if not quiet] or np.zeros((0, 2)))
        dq = np.array([(x, y) for x, y, quiet in r['deaths'] if quiet] or np.zeros((0, 2)))
        if len(d):
            ax.scatter(d[:, 0], d[:, 1], marker='x', s=40, color='#2a78d6', label='death in a fight')
        if len(dq):
            ax.scatter(dq[:, 0], dq[:, 1], marker='X', s=80, color='#d62a2a', label='death with nobody around')
        ax.set_title(f"{name}: {len(r['deaths'])} deaths in {len(seeds)} matches", color=MUTED, fontsize=10, loc='left')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.legend(frameon=False, fontsize=7, labelcolor=MUTED, loc='lower left')
    fig.suptitle('Where the players die and where the map hurts them (test matches, base graphics)', color=INK,
                 fontsize=12, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(a.out, dpi=120, facecolor=SURFACE)
    with open(a.out.replace('.png', '.json'), 'w') as fh:
        json.dump({n: r for n, r in runs}, fh)
    print(a.out)
