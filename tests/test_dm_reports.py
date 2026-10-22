from doomagent.dm.reports import d0_figure, gate_a, match_table_md, teacher_figure


def ci(m):
    return {'mean': m, 'lo': m - 0.5, 'hi': m + 0.5}


def summary(frags, kd_total, bot_median, stuck):
    return {'frags': ci(frags), 'reward': ci(frags), 'deaths': ci(4.0), 'kills': ci(frags + 1),
            'suicides': ci(1.0), 'kd': ci(kd_total), 'kd_total': kd_total, 'hits_per_shot': ci(0.4),
            'items': ci(3.0), 'path': ci(5000.0), 'stuck_share': ci(stuck), 'enemy_share': ci(0.2),
            'bot_median': ci(bot_median)}


def test_gate_a_rules():
    ok = gate_a({'teacher': summary(6.0, 1.4, 1.5, 0.02)})
    assert [p for _, _, p in ok] == [True, True, True]
    bad = gate_a({'teacher': summary(6.0, 0.9, 1.5, 0.02)})
    assert [p for _, _, p in bad] == [False, True, True]


def test_teacher_figure_and_table(tmp_path):
    report = {'teacher': summary(6.0, 1.4, 1.5, 0.02), 'random': summary(-15.0, -0.8, 0.5, 0.3)}
    out = tmp_path / 't.png'
    teacher_figure(report, {'combat': 0.3, 'chase': 0.1, 'items': 0.1, 'patrol': 0.5}, str(out))
    assert out.stat().st_size > 10_000
    assert match_table_md(report).count('\n') == 4


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
