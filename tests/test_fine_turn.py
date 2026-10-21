import pytest

from doomagent.config import FINE_LEFT, FINE_RIGHT, TURN_LEFT, Cfg

pytestmark = pytest.mark.vizdoom


def _angle_change(e, action):
    a0 = e.privileged().angle
    e.step(action)
    return (e.privileged().angle - a0 + 180.0) % 360.0 - 180.0


@pytest.mark.parametrize('mode', ['binary', 'delta'])
def test_fine_turns_turn_the_right_way_and_less_than_a_coarse_turn(mode):
    from doomagent.env import DoomEnv
    e = DoomEnv(Cfg(turn_mode=mode))
    e.reset(5000)
    coarse = _angle_change(e, TURN_LEFT)
    fine_left, fine_right = _angle_change(e, FINE_LEFT), _angle_change(e, FINE_RIGHT)
    e.close()
    assert fine_left > 0 > fine_right
    assert abs(fine_left) < abs(coarse)
