import os

import numpy as np
import pytest

from doomagent.config import SHOOT, TURN_LEFT, TURN_RIGHT, Cfg

pytestmark = pytest.mark.vizdoom


@pytest.fixture(scope='module')
def env():
    from doomagent.env import DoomEnv
    e = DoomEnv()
    yield e
    e.close()


def test_reset_shape_and_padding(env):
    obs = env.reset(0)
    assert obs.shape == (4, 120, 160) and obs.dtype == np.uint8
    assert all((obs[0] == obs[i]).all() for i in range(1, 4))


def test_render_and_privileged_shapes(env):
    env.reset(0)
    assert env.screen().shape == (480, 640, 3)
    assert env.privileged().labels_buffer.shape == (480, 640)


def test_network_frame_is_the_downscaled_render(env):
    from doomagent.env import to_net
    obs = env.reset(3)
    assert (obs[-1] == to_net(env.screen(), Cfg())).all()
    obs, _, _ = env.step(TURN_RIGHT)
    assert (obs[-1] == to_net(env.screen(), Cfg())).all()


def test_reset_is_deterministic_per_seed(env):
    assert (env.reset(7) == env.reset(7)).all()


def test_step_returns_pixels_only(env):
    env.reset(0)
    out = env.step(TURN_RIGHT)
    assert len(out) == 3
    obs, reward, done = out
    assert obs.shape == (4, 120, 160)
    assert isinstance(reward, float) and isinstance(done, bool)


def test_episode_terminates(env):
    env.reset(0)
    done, n = False, 0
    while not done and n < 3000:
        _, _, done = env.step(SHOOT)
        n += 1
    assert done


def _angle_change(e, action):
    a0 = e.privileged().angle
    e.step(action)
    return (e.privileged().angle - a0 + 180.0) % 360.0 - 180.0


@pytest.mark.parametrize('mode', ['binary', 'delta'])
def test_turn_directions(mode):
    # Doom angles grow counterclockwise: turning left increases ANGLE
    from doomagent.env import DoomEnv
    e = DoomEnv(Cfg(turn_mode=mode))
    e.reset(5000)
    left, right = _angle_change(e, TURN_LEFT), _angle_change(e, TURN_RIGHT)
    e.close()
    assert left > 0 > right


@pytest.mark.skipif(os.name != 'nt', reason='Windows power throttling only')
def test_env_unthrottles_its_vizdoom_process():
    # Windows 11 stops honoring the 1 ms timer for windowless background processes after a few
    # seconds, and every ViZDoom step then waits on the 15.6 ms timer (5 ms -> 100 ms per step)
    from doomagent.env import DoomEnv
    e = DoomEnv(Cfg())
    try:
        assert e.unthrottled
    finally:
        e.close()
