import numpy as np
import pytest

from doomagent.config import FINE_LEFT, FINE_RIGHT, N_ACTIONS, SHOOT, TURN_LEFT, TURN_RIGHT
from doomagent.record import noisy


def test_noisy_never_injects_shoot():
    f = noisy(1.0, np.random.default_rng(0))
    assert all(f(None, None, TURN_RIGHT) != SHOOT for _ in range(200))


def test_noisy_keeps_teacher_shots():
    f = noisy(1.0, np.random.default_rng(0))
    assert all(f(None, None, SHOOT) == SHOOT for _ in range(50))


def test_noisy_without_noise_is_the_label():
    f = noisy(0.0, np.random.default_rng(0))
    assert [f(None, None, a) for a in range(N_ACTIONS)] == [0, 1, 2, 3, 4]


def test_noise_draws_coarse_and_fine_turns():
    # S2 ruling: the random turn covers all four turn actions, so the data shows fine-turn recovery too
    f = noisy(1.0, np.random.default_rng(0))
    assert {f(None, None, TURN_RIGHT) for _ in range(400)} == {TURN_LEFT, TURN_RIGHT,
                                                               FINE_LEFT, FINE_RIGHT}


@pytest.mark.vizdoom
def test_labels_come_from_teacher_not_executor(tmp_path):
    from doomagent.env import DoomEnv
    from doomagent.record import episode_prefix, record_episode
    from doomagent.teacher import Teacher
    from doomagent.variants import BASE
    env = DoomEnv()
    pre = episode_prefix(str(tmp_path), BASE, 5000)
    n = record_episode(env, 5000, lambda o, p, label: TURN_LEFT, Teacher(), pre)
    env.close()
    z = np.load(pre + '.npz')
    frames, masks = np.load(pre + '.frames.npy'), np.load(pre + '.masks.npy')
    assert frames.shape == (n, 120, 160) and frames.dtype == np.uint8
    assert masks.shape == (n, 120, 20)
    assert set(z['actions'].tolist()) != {TURN_LEFT}
    assert float(np.abs(z['off']).max()) <= 1.0
    assert z['valid'].dtype == bool and str(z['variant']) == 'base'
    # wherever the enemy is visible, the mask is not empty
    vis = z['vis'].astype(bool)
    assert np.unpackbits(masks[vis], axis=-1).reshape(vis.sum(), -1).any(axis=1).mean() > 0.9
