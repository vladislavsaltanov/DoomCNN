import json


from dmfiles import make_match
from doomagent.dm.reports import (curves_figure, d0_figure, dataset_figure, dataset_stats, e0_figure, gate_a,
                                  gate_a2, match_table_md, teacher_figure, teacher_figure2)


def test_dataset_stats_counts(tmp_path):
    for s in range(3):
        make_match(tmp_path / 'd', s, T=40, first_at=(0, 20), variant='base' if s else 'tex_train_00')
    st = dataset_stats([str(tmp_path / 'd')])
    assert st['matches'] == 3 and st['steps'] == 120 and st['lives_per_match'] == [2, 2, 2]
    assert abs(sum(st['action_share']) - 1) < 1e-9 and st['action_share'][0] == 0.1
    assert st['mode_share'][0] == 1.0 and st['noise_share'] == 0.0
    assert st['per_variant'] == {'base': 2, 'tex_train_00': 1} and st['gb'] > 0


def test_dataset_figure_writes_png(tmp_path):
    for s in range(3):
        make_match(tmp_path / 'd', s, T=40, first_at=(0, 20))
    out = tmp_path / 'ds.png'
    dataset_figure(dataset_stats([str(tmp_path / 'd')]), str(out))
    assert out.stat().st_size > 10_000


def test_curves_figure_writes_png(tmp_path):
    logs = {}
    for name in ('m0', 'm1'):
        rows = [dict(epoch=e, holdout_ce=2.0 - 0.1 * e, holdout_acc=0.2 + 0.03 * e, val_score=0.1 * e,
                     frags=dict(mean=0.1 * e, lo=0.1 * e - 0.5, hi=0.1 * e + 0.5)) for e in (5, 10, 15)]
        p = tmp_path / f'{name}.log.json'
        p.write_text(json.dumps(dict(config={}, log=rows)))
        logs[name] = str(p)
    conf = [[10 if i == j else (3 if j == 0 else 1) for j in range(10)] for i in range(10)]
    (tmp_path / 'm1.live.json').write_text(json.dumps(dict(confusion=conf, holdout_confusion=conf)))
    out = tmp_path / 'c.png'
    curves_figure(logs, str(out), teacher=ci(6.25), confusion_of=str(tmp_path / 'm1.live.json'),
                  action_names=[str(i) for i in range(10)])
    assert out.stat().st_size > 10_000


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


def test_e0_figure_writes_png(tmp_path):
    turns = {d: d * 0.99 for d in (1.0, -1.0, 3.0, -3.0, 10.0, -10.0, 40.0, -40.0)}
    moves = {'forward': 29.5, 'back': 24.0, 'strafe_left': 23.0, 'strafe_right': 23.0, 'forward_left': 33.0}
    bots = {'player': ci(-3.0), 'bot_median': ci(3.5), 'bot_best': ci(6.0)}
    out = tmp_path / 'e0.png'
    e0_figure(turns, moves, bots, str(out))
    assert out.stat().st_size > 10_000


def summary2(frags, kd_total, best, first):
    s = summary(frags, kd_total, 1.0, 0.0)
    s.update(bot_best=ci(best), first_share=first, weapon_share={k: 1 / 7 for k in range(1, 8)})
    return s


def test_gate_a2_rules():
    assert [p for _, _, p in gate_a2({'teacher': summary2(20.0, 3.5, 9.0, 0.95)})] == [True, True, True]
    assert [p for _, _, p in gate_a2({'teacher': summary2(12.0, 2.0, 9.0, 0.6)})] == [False, False, False]


def test_teacher_figure2_writes_png(tmp_path):
    diag = {'death_cause': {'bot seen': 0.6, 'unseen': 0.3, 'suicide': 0.1},
            'hit_share_by_weapon': {'pistol': 0.4, 'chaingun': 0.3, 'rocket launcher': 0.5}}
    out = tmp_path / 't2.png'
    teacher_figure2({'teacher': summary2(15.0, 3.0, 8.0, 0.9)}, diag, str(out))
    assert out.stat().st_size > 10_000
