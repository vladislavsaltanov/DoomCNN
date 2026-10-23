"""Watch or record a deathmatch player on cig.wad MAP01.

    python scripts/dm_play.py --who random --seed 35000 --boxes --hud
    python scripts/dm_play.py --who random --seed 35000 --boxes --hud --no-window --record artifacts/d0_random.mp4

One frame per game tic (35 per second). Boxes, the status bar and the action list are drawn for the
viewer only. --who takes random, teacher or a dm_*.pt checkpoint (M0 or M1).
"""
import argparse
import os
import time

import cv2
import numpy as np

from doomagent.dm.config import DM, N_ACTIONS, VARIANTS_DIR
from doomagent.dm.env import DeathmatchEnv
from doomagent.dm.viewer import compose_view
from doomagent.variants import get_variant
from doomagent.viz import VideoSink


class Show:
    # writes every tic to the video and paces the window at 35 tics per second, like viz.play_episode
    def __init__(self, sink, window, speed):
        self.sink, self.window, self.tic_s, self.due = sink, window, 1.0 / (35 * speed), None

    def __call__(self, frame):
        if self.sink is not None:
            self.sink.write(frame)
        if not self.window:
            return
        cv2.imshow('DoomCNN deathmatch', frame)
        now = time.perf_counter()
        if self.due is None or now - self.due > 0.1:  # first frame, or too late after a pause
            self.due = now
        self.due += self.tic_s
        key = cv2.waitKey(max(1, int((self.due - now) * 1000))) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt
        if key == ord(' '):
            cv2.waitKey(0)
            self.due = None


class TeacherChooser:
    def __init__(self, cfg):
        from doomagent.dm.teacher import DMTeacher
        self.teacher = DMTeacher(cfg)

    def reset(self):
        self.teacher.reset()

    def __call__(self, obs, priv):
        return self.teacher(obs, priv), None


def make_chooser(who, seed):
    # chooser(obs, priv) -> (action, info for the overlay)
    if who == 'random':
        rng = np.random.default_rng(seed)
        return lambda obs, priv: (int(rng.integers(0, N_ACTIONS)), None)
    if who == 'teacher':
        return TeacherChooser(DM)
    return ModelChooser(who)


class ModelChooser:
    # a trained M0 or M1 on the CPU: it gets frames, vec and first only; probs go to the overlay
    def __init__(self, path):
        from doomagent.dm.agent import DMAgent
        from doomagent.dm.model import load_model
        self.agent = DMAgent(load_model(path, DM))

    def reset(self):
        self.agent.reset()

    def __call__(self, obs, priv):
        info = self.agent.explain(obs.frames, obs.vec, obs.first)
        return info['action'], info


def snapshot_grid(snaps, path):
    # one frame per teacher mode, 2x2, half size
    tiles = [cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2), interpolation=cv2.INTER_AREA)
             for f in snaps.values()]
    for t, mode in zip(tiles, snaps):
        cv2.putText(t, mode, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (71, 181, 255), 2, cv2.LINE_AA)
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])]))


ap = argparse.ArgumentParser()
ap.add_argument('--who', default='random', help='random | teacher | path to a model')
ap.add_argument('--variant', default='base')
ap.add_argument('--seed', type=int, default=35000)
ap.add_argument('--matches', type=int, default=1)
ap.add_argument('--record', help='write an mp4 here')
ap.add_argument('--no-window', action='store_true')
ap.add_argument('--boxes', action='store_true', help='draw label boxes for the viewer')
ap.add_argument('--hud', action='store_true', help='draw the Doom status bar for the viewer')
ap.add_argument('--speed', type=float, default=1.0)
ap.add_argument('--map', action='store_true', help='minimap and the teacher state on the right')
ap.add_argument('--snapshots', help='PNG: one frame per teacher mode (with --who teacher)')
a = ap.parse_args()
cfg = DM
env = DeathmatchEnv(cfg, get_variant(a.variant, VARIANTS_DIR))
hud = None
if a.hud:
    from doomagent.hud import Hud
    hud = Hud()
sink = None
if a.record:
    os.makedirs(os.path.dirname(a.record) or '.', exist_ok=True)
    sink = VideoSink(a.record, 35, (cfg.render_w + (320 if a.map else 0), cfg.render_h))
show = Show(sink, not a.no_window, a.speed)
choose = make_chooser(a.who, a.seed)
teacher = getattr(choose, 'teacher', None)
mapview, snaps, last = None, {}, {}


def tic(rgb, p, action, info):
    last['frame'] = compose_view(rgb, p, action, info, cfg, a.boxes, hud, mapview, teacher)
    show(last['frame'])


try:
    for i in range(a.matches):
        seed = a.seed + i
        obs, done, steps = env.reset(seed), False, 0
        if a.map and mapview is None:
            from doomagent.dm.minimap import MapView
            mapview = MapView(env.sectors)
        if hasattr(choose, 'reset'):
            choose.reset()
        while not done:
            action, info = choose(obs, env.privileged())
            obs, _, done = env.step(action, on_tic=lambda rgb, p: tic(rgb, p, action, info))
            steps += 1
            if a.snapshots and teacher is not None and steps > 100 and 'frame' in last \
                    and teacher.mode not in snaps:
                snaps[teacher.mode] = last['frame'].copy()
        p = env.privileged()
        print(f'variant={a.variant} seed={seed} frags={p.frags:.0f} deaths={p.deaths:.0f} steps={steps}')
    if a.snapshots and snaps:
        snapshot_grid(snaps, a.snapshots)
        print('snapshots:', list(snaps), '->', a.snapshots)
except KeyboardInterrupt:
    pass
finally:
    if sink is not None:
        sink.close()
    env.close()
    cv2.destroyAllWindows()
