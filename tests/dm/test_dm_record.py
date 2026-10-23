from dataclasses import replace

import numpy as np
import pytest

from dmfiles import FakeMatchEnv, StubTeacher, row
from doomagent.dm.actions import FORWARD, HEAD_SIZES, STAY, VEC_DIM, turn_index
from doomagent.dm.config import DM
from doomagent.dm.record import MatchNoise, record_match
from doomagent.variants import BASE

LABEL = (FORWARD, turn_index(10.0), 1, 0)


def test_noise_changes_one_move_or_turn_head():
    f = MatchNoise(0.1)
    f.reset(0)
    out = [f(None, None, LABEL) for _ in range(2000)]
    changed = [a for a in out if a != LABEL]
    assert abs(len(changed) / 2000 - 0.1) < 0.02
    assert all(sum(x != y for x, y in zip(a[:2], LABEL[:2])) == 1 and a[2:] == LABEL[2:] for a in changed)
    assert all(0 <= v < k for a in changed for v, k in zip(a, HEAD_SIZES))
    assert {a[0] != LABEL[0] for a in changed} == {True, False}  # both heads get noise


def test_match_noise_reset_repeats():
    f = MatchNoise(0.5)
    f.reset(7)
    a = [f(None, None, LABEL) for _ in range(50)]
    f.reset(7)
    assert a == [f(None, None, LABEL) for _ in range(50)]


def test_labels_come_from_the_teacher(tmp_path):
    env = FakeMatchEnv([row(first=i == 0) for i in range(12)])
    label = (STAY, turn_index(-3.0), 0, 1)
    n = record_match(env, 1, lambda o, p, lab: (FORWARD, 0, 0, 0), StubTeacher(label), str(tmp_path / 'ep_base_1'))
    z = np.load(tmp_path / 'ep_base_1.npz')
    assert n == 12 and z['actions'].shape == (12, 4) and (z['actions'] == label).all()
    assert (z['executed'] == (FORWARD, 0, 0, 0)).all() and z['first'][0] and z['vec'].shape == (12, VEC_DIM)
    assert z['weapon'].shape == (12,)
    assert np.load(tmp_path / 'ep_base_1.frames.npy').shape == (12, DM.height, DM.width)


def test_match_end_during_death_keeps_arrays_aligned(tmp_path):
    rows = [row(first=i == 0) for i in range(9)] + [row(deaths=1, done=True)]
    n = record_match(FakeMatchEnv(rows), 1, lambda o, p, lab: lab, StubTeacher(LABEL), str(tmp_path / 'ep_base_1'))
    z = np.load(tmp_path / 'ep_base_1.npz')
    assert all(len(z[k]) == n for k in ('actions', 'executed', 'vec', 'first', 'rewards', 'modes', 'pose', 'weapon'))


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
