from types import SimpleNamespace as NS

import cv2
import numpy as np

from doomagent.viz import draw_overlay, play_episode, stack_image, video_frame_count


class FakeEnv:
    def __init__(self, n):
        self.n, self.t = n, 0

    def reset(self, seed):
        self.t = 0
        return np.zeros((4, 120, 160), np.uint8)

    def privileged(self):
        return None

    def screen(self):
        return np.full((480, 640, 3), 30 * self.t, np.uint8)

    def step(self, a):
        self.t += 1
        return np.full((4, 120, 160), 40 * self.t, np.uint8), 0.0, self.t >= self.n


def test_stack_image_size(tmp_path):
    stack_image(np.zeros((4, 120, 160), np.uint8), str(tmp_path / 's.png'), scale=1)
    assert cv2.imread(str(tmp_path / 's.png'), cv2.IMREAD_GRAYSCALE).shape == (120, 640)


def test_overlay_draws_on_full_frame():
    info = dict(probs=np.array([0.05, 0.1, 0.5, 0.2, 0.15]), vis=0.9, off=0.25,
                seg=np.zeros((11, 16), np.float32))
    box = NS(x=10, y=10, width=20, height=30)
    img = draw_overlay(np.zeros((480, 640, 3), np.uint8), 2, info, [box],
                       np.zeros((120, 160), np.uint8))
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8 and img.any()


def test_play_episode_writes_one_frame_per_step(tmp_path):
    from doomagent.viz import VideoSink
    out = str(tmp_path / 'ep.mp4')
    sink = VideoSink(out, 8.75, (640, 480))
    total, n = play_episode(FakeEnv(5), lambda obs, priv: (2, None), 0, sink)
    sink.close()
    assert n == 5 and video_frame_count(out) == 5
