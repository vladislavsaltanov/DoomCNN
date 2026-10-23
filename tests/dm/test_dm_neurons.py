import numpy as np

from dmfiles import Poison, StubTeacher
from doomagent.dm.config import MODES, TURN_LEFT
from doomagent.dm.env import DMObs
from doomagent.dm.neurons import CONDITIONS, ShadowChooser, _recent, place_maps, selectivity, state_map
from doomagent.dm.reports import place_figure, selectivity_figure, states_figure


def test_shadow_chooser_keeps_the_student_on_pixels():
    seen = []

    class Spy:
        def reset(self):
            pass

        def __call__(self, obs, priv):
            seen.append(priv)
            return 0

    c = ShadowChooser(Spy(), StubTeacher(TURN_LEFT))
    c(DMObs(np.zeros((4, 120, 160), np.uint8), np.zeros(2, np.float32), True), Poison())
    assert seen == [None] and c.take_trace()['modes'].tolist() == [MODES.index('patrol')]


def left_half_acts():
    g = np.random.default_rng(0)
    px, py = g.uniform(0, 1024, 5000), g.uniform(0, 1024, 5000)
    act = g.normal(0, 0.1, (5000, 512))
    act[:, 7] += (px < 512) * 2.0
    return dict(act=act, px=px, py=py)


def test_place_map_finds_a_left_half_unit():
    unit, grid, _ = place_maps(left_half_acts(), top=3)[0]
    assert unit == 7 and np.nanmean(grid[:, :8]) > np.nanmean(grid[:, 8:]) + 1.0


def synthetic(n=4000):
    g = np.random.default_rng(0)
    acts = {k: g.random(n) < 0.3 for k in ('bot_visible', 'bot_recent', 'health_low')}
    acts['mode'] = g.integers(0, 4, n)
    acts['act'] = g.normal(0, 1, (n, 512))
    acts['act'][:, 3] += 3 * acts['bot_visible']
    return acts


def test_selectivity_picks_the_correlated_condition():
    best, strength, corr = selectivity(synthetic())
    assert best[3] == 0 and strength[3] > 0.5 and corr.shape == (512, len(CONDITIONS))


def test_recent_is_after_a_sighting_within_the_life():
    enemy = np.array([1, 0, 0, 0, 0, 0], bool)
    first = np.array([1, 0, 0, 1, 0, 0], bool)
    assert _recent(enemy, first, k=5).tolist() == [False, True, True, False, False, False]


def test_state_map_shapes():
    g = np.random.default_rng(0)
    xy, mode = state_map(dict(act=g.normal(0, 1, (400, 512)), mode=g.integers(0, 4, 400)), n=300)
    assert xy.shape == (300, 2) and mode.shape == (300,)


def test_figures_write_png(tmp_path):
    sectors = [(0.0, 128.0, ((0.0, 0.0, 1024.0, 0.0, True), (1024.0, 0.0, 1024.0, 1024.0, True),
                             (1024.0, 1024.0, 0.0, 1024.0, True), (0.0, 1024.0, 0.0, 0.0, True)))]
    place_figure(place_maps(left_half_acts()), sectors, tmp_path / 'p.png')
    best, strength, _ = selectivity(synthetic())
    selectivity_figure(best, strength, CONDITIONS, tmp_path / 's.png')
    g = np.random.default_rng(0)
    states_figure(g.normal(0, 1, (300, 2)), g.integers(0, 4, 300), tmp_path / 't.png')
    for f in ('p', 's', 't'):
        assert (tmp_path / f'{f}.png').stat().st_size > 10_000
