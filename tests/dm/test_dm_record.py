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


def test_live_tap_publishes_the_current_step(tmp_path):
    # a recording worker shows what it plays: frame, minimap and state, at most every every_s seconds
    import json
    from doomagent.dm.livetap import LiveTap
    tap = LiveTap(str(tmp_path), every_s=0.0, worker='w1')
    env = FakeMatchEnv([row(first=i == 0, frags=float(i // 4)) for i in range(12)])
    record_match(env, 5, lambda o, p, lab: lab, StubTeacher(LABEL), str(tmp_path / 'ep_base_5'), live=tap)
    st = json.load(open(tmp_path / 'w1.json'))
    assert st['seed'] == 5 and st['step'] == 11 and st['frags'] == 2.0 and st['mode'] == 'patrol'
    assert st['variant'] == 'base' and 'fire' in st['label']
    assert (tmp_path / 'w1.screen.jpg').stat().st_size > 0


def test_live_tap_waits_between_frames(tmp_path):
    from doomagent.dm.livetap import LiveTap
    tap = LiveTap(str(tmp_path), every_s=3600.0, worker='w1')
    env = FakeMatchEnv([row(first=i == 0) for i in range(6)])
    record_match(env, 5, lambda o, p, lab: lab, StubTeacher(LABEL), str(tmp_path / 'ep_base_5'), live=tap)
    import json
    assert json.load(open(tmp_path / 'w1.json'))['step'] == 0  # only the first step was published


def test_bot_visible_ignores_hidden_bots(tmp_path):
    # a label with no pixels on screen (behind a wall) is not a visible bot in the dataset
    from dataclasses import replace as rep
    class Hidden(FakeMatchEnv):
        def privileged(self):
            return rep(self._priv(), vis_px={7: 0})
    env = Hidden([row(first=i == 0, bot=True) for i in range(5)])
    record_match(env, 1, lambda o, p, lab: lab, StubTeacher(LABEL), str(tmp_path / 'ep_base_1'))
    assert not np.load(tmp_path / 'ep_base_1.npz')['bot_visible'].any()
    record_match(FakeMatchEnv([row(first=i == 0, bot=True) for i in range(5)]), 1, lambda o, p, lab: lab,
                 StubTeacher(LABEL), str(tmp_path / 'ep_base_2'))
    assert np.load(tmp_path / 'ep_base_2.npz')['bot_visible'].all()  # unknown pixel counts pass


def test_live_tap_lists_finished_matches(tmp_path):
    import json
    from doomagent.dm.livetap import LiveTap
    tap = LiveTap(str(tmp_path), worker='w1')
    tap.match_done(5, frags=7.0, deaths=2.0, best_bot=6.0, first=True)
    tap.match_done(6, frags=3.0, deaths=4.0, best_bot=5.0, first=False)
    done = json.load(open(tmp_path / 'w1.done.json'))
    assert [d['seed'] for d in done] == [5, 6] and done[0]['frags'] == 7.0 and done[1]['first'] is False
