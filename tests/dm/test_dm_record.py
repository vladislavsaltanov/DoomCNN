from dataclasses import replace

import numpy as np
import pytest

from dmfiles import FakeMatchEnv, StubTeacher, row
from doomagent.dm.config import DM, FORWARD, NOISE_ACTIONS, SHOOT, TURN_LEFT
from doomagent.dm.record import MatchNoise, record_match
from doomagent.variants import BASE


def test_match_noise_never_injects_or_replaces_shoot():
    f = MatchNoise(1.0)
    f.reset(0)
    assert all(f(None, None, SHOOT) == SHOOT for _ in range(50))
    assert all(f(None, None, FORWARD) != SHOOT for _ in range(300))


def test_match_noise_covers_every_move_and_turn():
    f = MatchNoise(1.0)
    f.reset(0)
    assert {f(None, None, FORWARD) for _ in range(600)} == set(NOISE_ACTIONS)


def test_match_noise_reset_repeats():
    f = MatchNoise(0.5)
    f.reset(7)
    a = [f(None, None, FORWARD) for _ in range(50)]
    f.reset(7)
    assert a == [f(None, None, FORWARD) for _ in range(50)]


def test_labels_come_from_the_teacher(tmp_path):
    env = FakeMatchEnv([row(first=i == 0) for i in range(12)])
    n = record_match(env, 1, lambda o, p, label: FORWARD, StubTeacher(TURN_LEFT), str(tmp_path / 'ep_base_1'))
    z = np.load(tmp_path / 'ep_base_1.npz')
    assert n == 12 and (z['actions'] == TURN_LEFT).all() and (z['executed'] == FORWARD).all()
    assert z['first'][0] and z['vec'].shape == (12, 2)
    assert np.load(tmp_path / 'ep_base_1.frames.npy').shape == (12, DM.height, DM.width)


def test_match_end_during_death_keeps_arrays_aligned(tmp_path):
    rows = [row(first=i == 0) for i in range(9)] + [row(deaths=1, done=True)]
    n = record_match(FakeMatchEnv(rows), 1, lambda o, p, label: FORWARD, StubTeacher(FORWARD),
                     str(tmp_path / 'ep_base_1'))
    z = np.load(tmp_path / 'ep_base_1.npz')
    assert all(len(z[k]) == n for k in ('actions', 'executed', 'vec', 'first', 'rewards', 'modes', 'pose'))


@pytest.mark.vizdoom
def test_parallel_record_matches_serial(tmp_path):
    from doomagent.dm.record import record_dataset
    cfg = replace(DM, timelimit=0.1)
    plan = [(BASE, (38000, 38001))]
    record_dataset(str(tmp_path / 's'), plan, MatchNoise(0.1), cfg)
    record_dataset(str(tmp_path / 'p'), plan, MatchNoise(0.1), cfg, workers=2)
    for s in (38000, 38001):
        a, b = (np.load(tmp_path / d / f'ep_base_{s}.npz') for d in 'sp')
        assert all(np.array_equal(a[k], b[k]) for k in a.files)
        assert np.array_equal(*(np.load(tmp_path / d / f'ep_base_{s}.frames.npy') for d in 'sp'))
    assert (tmp_path / 'p' / 'meta.json').read_text() == (tmp_path / 's' / 'meta.json').read_text()
