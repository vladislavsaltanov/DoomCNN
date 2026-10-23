import itertools

import numpy as np

from doomagent.dm.actions import (BACK_LEFT, BACK_RIGHT, FORWARD, HEAD_SIZES, HEADS, N_LOGITS, NOOP, STAY, TURN_DEG,
                                  TURN_MIRROR, VEC_DIM, buttons_for, make_vec, mirror_action, turn_index)
from doomagent.env import DELTA_SIGN


def test_heads_and_sizes():
    assert HEADS == ('move', 'turn', 'fire', 'weapon') and HEAD_SIZES == (9, 15, 2, 2)
    assert N_LOGITS == sum(HEAD_SIZES) == 28 and len(TURN_DEG) == 15 and NOOP == (STAY, 0, 0, 0)


def test_mirror_action_table():
    assert mirror_action((BACK_LEFT, TURN_DEG.index(-20.0), 1, 0)) == (BACK_RIGHT, TURN_DEG.index(20.0), 1, 0)
    assert all(mirror_action(mirror_action(a)) == a for a in itertools.product(*map(range, HEAD_SIZES)))


def test_turn_mirror_negates():
    assert all(TURN_DEG[TURN_MIRROR[i]] == -TURN_DEG[i] for i in range(15))


def test_turn_index():
    assert turn_index(-30.0) == TURN_DEG.index(-30.0) and turn_index(0.0) == 0


def test_make_vec():
    v = make_vec(150.0, 100.0, 5, 25.0)
    assert v.shape == (VEC_DIM,) == (10,) and v.dtype == np.float32
    assert np.allclose(v[:3], [1.5, 0.5, 0.5]) and v[3:].tolist() == [0, 0, 0, 0, 1, 0, 0]


def test_make_vec_fist_has_no_ammo():
    v = make_vec(100.0, 0.0, 1, 50.0)
    assert v[2] == 0.0 and v[3] == 1.0


def test_buttons_for_turn_and_weapon():
    first, rest = buttons_for((FORWARD, TURN_DEG.index(10.0), 1, 1), 4)
    assert first[0] == 1 and first[4] == 1 and first[5] == 1 and rest[5] == 0
    assert first[6] == rest[6] == -DELTA_SIGN * 2.5  # + is a left turn, the delta button turns right on +
