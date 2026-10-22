from doomagent.dm.reports import d0_figure


def test_d0_figure_writes_png(tmp_path):
    physics = {'forward': [7.1, 15.6, 21.4, 25.3, 27.9, 29.7] + [30.0] * 6,
               'strafe_left': [24.7, 22.0, 23.2] + [23.0] * 9,
               'turn_left': {'hold': [7.03, 12.3] + [14.06] * 10, 'alternate': [14.06, -14.06] * 6},
               'fine_left': {'hold': [2.02] * 12, 'alternate': [2.02, -2.02] * 6}}
    bots = {n: {'bot_median': {'mean': m, 'lo': m - 0.5, 'hi': m + 0.5}, 'ms_per_step': ms}
            for n, m, ms in ((3, 0.5, 5.1), (5, 1.0, 6.0), (7, 1.5, 7.7))}
    out = tmp_path / 'd0.png'
    d0_figure(physics, bots, str(out))
    assert out.exists() and out.stat().st_size > 10_000
