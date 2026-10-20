import argparse
import os

import cv2
import numpy as np

from doomagent.config import Cfg
from doomagent.env import DoomEnv
from doomagent.variants import get_variant
from doomagent.viz import VideoSink, play_episode

ap = argparse.ArgumentParser()
ap.add_argument('--who', default='teacher', help='teacher | random | path to .pt or .zip')
ap.add_argument('--variant', default='base')
ap.add_argument('--seed', type=int, default=5000)
ap.add_argument('--episodes', type=int, default=1)
ap.add_argument('--record', help='write an mp4 here')
ap.add_argument('--no-window', action='store_true')
ap.add_argument('--boxes', action='store_true', help='draw label boxes for the viewer')
ap.add_argument('--speed', type=float, default=1.0)
a = ap.parse_args()
cfg = Cfg()


def make_chooser():
    if a.who == 'teacher':
        from doomagent.teacher import Teacher
        teacher = Teacher(cfg)

        class TeacherChooser:
            def reset(self):
                teacher.reset()

            def __call__(self, obs, priv):
                return teacher(obs, priv), None

        return TeacherChooser()
    if a.who == 'random':
        rng = np.random.default_rng(a.seed)
        return lambda obs, priv: (int(rng.integers(0, 3)), None)
    from doomagent.agent import load_agent
    agent = load_agent(a.who, cfg)

    def choose(obs, priv):
        info = agent.explain(obs)  # the agent gets the pixel stack only
        return info['action'], info

    return choose


boxes_fn = None
if a.boxes:
    from doomagent.teacher import is_enemy_name
    boxes_fn = lambda priv: [lb for lb in priv.labels if is_enemy_name(lb.object_name)]

env = DoomEnv(cfg, get_variant(a.variant))
sink = None
if a.record:
    os.makedirs(os.path.dirname(a.record) or '.', exist_ok=True)
    sink = VideoSink(a.record, 35 / cfg.frame_skip, (cfg.render_w, cfg.render_h))
choose = make_chooser()
try:
    for i in range(a.episodes):
        total, n = play_episode(env, choose, a.seed + i, sink, not a.no_window, a.speed,
                                boxes_fn, cfg)
        print(f'variant={a.variant} seed={a.seed + i} reward={total:.1f} steps={n}')
except KeyboardInterrupt:
    pass
finally:
    if sink is not None:
        sink.close()
    env.close()
    cv2.destroyAllWindows()
