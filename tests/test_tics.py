import numpy as np
import pytest

from doomagent.viz import VideoSink, play_episode, video_frame_count


class TicFakeEnv:
    # every agent step runs frame_skip tics and reports each one through on_tic
    def __init__(self, n_steps):
        self.n_steps, self.steps, self.tics = n_steps, 0, 0

    def reset(self, seed):
        self.steps = self.tics = 0
        return np.zeros((4, 120, 160), np.uint8)

    def privileged(self):
        return None

    def screen(self):
        return np.full((480, 640, 3), 30 * (self.tics % 8), np.uint8)

    def step(self, action, on_tic=None):
        for _ in range(4):
            self.tics += 1
            if on_tic is not None:
                on_tic(self.screen(), None)
        self.steps += 1
        return np.full((4, 120, 160), 40 * self.steps, np.uint8), 0.0, self.steps >= self.n_steps


def test_every_tic_writes_frame_skip_frames_per_step(tmp_path):
    out = str(tmp_path / 'tics.mp4')
    sink = VideoSink(out, 35, (640, 480))
    total, n = play_episode(TicFakeEnv(3), lambda obs, priv: (2, None), 0, sink, every_tic=True)
    sink.close()
    assert n == 3 and video_frame_count(out) == 12


@pytest.mark.vizdoom
def test_tic_by_tic_step_matches_the_frame_skip_step():
    from doomagent.env import DoomEnv
    fast, slow = DoomEnv(), DoomEnv()
    fast.reset(5000)
    slow.reset(5000)
    rng = np.random.default_rng(0)
    for _ in range(40):
        action = int(rng.integers(0, 3))
        tics = []
        obs_a, r_a, done_a = fast.step(action)
        obs_b, r_b, done_b = slow.step(action, on_tic=lambda rgb, priv: tics.append(rgb.copy()))
        assert (obs_a == obs_b).all() and r_a == r_b and done_a == done_b
        if not done_b:
            assert len(tics) == 4
        else:
            break
    fast.close()
    slow.close()
