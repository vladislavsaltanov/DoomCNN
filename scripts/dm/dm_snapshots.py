"""PNG snapshots of the deathmatch work for the slice reports.

    python scripts/dm/dm_snapshots.py stack --out artifacts/dm/d0_stack.png
    python scripts/dm/dm_snapshots.py view --out artifacts/dm/d0_view.png
    python scripts/dm/dm_snapshots.py brain runs/dm/m1_pilot.pt --at 300 --out artifacts/dm/d2_brain_m1.png
    python scripts/dm/dm_snapshots.py memory runs/dm/m1_pilot.pt --at 400 --out artifacts/dm/d2_memory_m1.png
"""
import argparse
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from doomagent.dm.actions import FORWARD, HEAD_SIZES, STAY, turn_index  # noqa: E402
from doomagent.dm.config import DM, VAL_SEEDS  # noqa: E402
from doomagent.dm.env import DeathmatchEnv  # noqa: E402
from doomagent.dm.viewer import compose_view, label_class  # noqa: E402
from doomagent.reports import INK, MUTED, SURFACE  # noqa: E402


def stack(out, seed):
    # what the network gets: the stack at a match start, in the middle of a life, after a respawn
    env = DeathmatchEnv(DM)
    rows = [('match start: four copies of the first frame', env.reset(seed).frames)]
    for i in range(12):
        obs, _, _ = env.step((STAY, turn_index(10.0), 0, 0) if i % 2 else (FORWARD, 0, 0, 0))
    rows.append(('middle of a life: the last four steps', obs.frames))
    env.game.send_game_command('kill')
    obs, _, _ = env.step((FORWARD, 0, 0, 0))
    assert obs.first, 'the kill did not start a new life'
    rows.append(('right after a respawn: four copies of the new first frame', obs.frames))
    env.close()
    fig, ax = plt.subplots(3, 4, figsize=(10.4, 6.9), facecolor=SURFACE,
                           gridspec_kw=dict(wspace=0.04, hspace=0.5, left=0.02, right=0.98,
                                            top=0.9, bottom=0.02))
    for r, (title, frames) in enumerate(rows):
        for c in range(4):
            a = ax[r, c]
            a.imshow(frames[c], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            a.set_xticks([])
            a.set_yticks([])
            a.set_title('t (newest)' if c == 3 else f't-{3 - c}', fontsize=8, color=MUTED)
        ax[r, 0].annotate(title, xy=(0, 1.22), xycoords='axes fraction', fontsize=10, color=INK)
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def view(out, seed):
    # one 640x480 frame with a bot, an item and a barrel boxed if a random walk finds one
    from doomagent.hud import Hud
    env = DeathmatchEnv(DM)
    rng = np.random.default_rng(1)
    _, done, best = env.reset(seed), False, None
    while not done:
        p = env.privileged()
        kinds = {label_class(lb.object_name) for lb in p.labels} - {None}
        score = 100 * len(kinds) + len(p.labels)
        if best is None or score > best[0]:
            best = (score, env.screen().copy(), p)
        if len(kinds) == 3:
            break
        _, _, done = env.step(tuple(int(rng.integers(0, k)) for k in HEAD_SIZES))
    env.close()
    cv2.imwrite(out, compose_view(best[1], best[2], None, None, DM, True, Hud()))


def play_to(model_path, seed, at):
    # the model plays its own match up to step `at`; the network view and the memory tape of that step
    import torch
    from doomagent.dm.agent import DMAgent
    from doomagent.dm.brain import trace
    from doomagent.dm.memory_view import MemoryTape, events_from
    from doomagent.dm.model import load_model
    agent = DMAgent(load_model(model_path, DM))
    env, tape = DeathmatchEnv(DM), MemoryTape()
    obs, prev = env.reset(seed), None
    for step in range(at + 1):
        priv = env.privileged()
        info = agent.explain(obs.frames, obs.vec, obs.first)
        if info['h'] is not None:
            tape.push(info['h'], info['z'], events_from(prev, priv, obs.first, step == 0))
        if step == at:
            h = None if info['h_prev'] is None else torch.from_numpy(info['h_prev']).reshape(1, 1, 512)
            t = trace(agent.model, obs.frames, obs.vec, h)
            break
        prev = priv
        obs, _, done = env.step(info['action'])
        assert not done, f'the match ended before step {at}'
    env.close()
    return t, obs, tape


def brain(out, seed, model, at):
    from doomagent.dm.brain import render
    t, obs, _ = play_to(model, seed, at)
    cv2.imwrite(out, render(t, obs.frames, obs.vec))


def memory(out, seed, model, at):
    _, _, tape = play_to(model, seed, at)
    assert tape.cols, f'{model} has no memory (M0): the tape is only for M1'
    cv2.imwrite(out, tape.render())


ap = argparse.ArgumentParser()
ap.add_argument('what', choices=['stack', 'view', 'brain', 'memory'])
ap.add_argument('model', nargs='?', help='brain, memory: a dm_*.pt checkpoint')
ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=VAL_SEEDS[0])
ap.add_argument('--at', type=int, default=300, help='brain, memory: the step to draw')
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
if a.what in ('brain', 'memory'):
    {'brain': brain, 'memory': memory}[a.what](a.out, a.seed, a.model, a.at)
else:
    {'stack': stack, 'view': view}[a.what](a.out, a.seed)
print('written', a.out)
