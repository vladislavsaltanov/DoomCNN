import json

import pytest

from epfiles import make_ep

from doomagent.reports import (dataset_report, dataset_stats, mask_preview, plot_bc_curves,
                               report_table_md)


def summary(mean):
    return dict(reward=dict(mean=mean, lo=mean - 1, hi=mean + 1),
                kills=dict(mean=mean, lo=0, hi=1), shots=dict(mean=20.0, lo=0, hi=1),
                length=dict(mean=300.0, lo=0, hi=1), kills_per_shot=0.9, idle_rate=0.0,
                pressed_per_ep=25.0)


def test_table_has_header_rows_and_dashes_for_teacher():
    rep = {'teacher@id': summary(24.0),
           'bc@id': dict(summary(22.0), vs_teacher=(-2.0, -3.0, -1.0), latency_ms=1.5)}
    lines = report_table_md(rep).strip().split('\n')
    assert lines[0].startswith('| model') and len(lines) == 4
    assert '| teacher@id | 24.00 [23.00, 25.00] | -' in lines[2]
    assert '-2.00 [-3.00, -1.00]' in lines[3] and '1.5' in lines[3]


def test_dataset_stats_counts(tmp_path):
    make_ep(tmp_path, 1)
    make_ep(tmp_path, 2, variant='tex_train_00')
    s = dataset_stats([str(tmp_path)])
    assert s['episodes'] == 2 and s['steps'] == 12
    assert s['action_share'] == [1 / 3, 1 / 3, 1 / 3, 0.0, 0.0]
    assert s['vis_share'] == 0.0 and abs(s['valid_share'] - 5 / 6) < 1e-9
    assert s['variants'] == {'base': 1, 'tex_train_00': 1}


def test_dataset_report_and_mask_preview_write_files(tmp_path):
    make_ep(tmp_path, 1, vis=1)
    dataset_report([str(tmp_path)], str(tmp_path / 'd.png'), str(tmp_path / 'd.md'))
    mask_preview([str(tmp_path)], str(tmp_path / 'm.png'), n=2)
    assert all((tmp_path / f).exists() for f in ('d.png', 'd.md', 'm.png'))


def test_dataset_stats_action_share_on_valid_steps(tmp_path):
    # the action loss sees only steps with ammo, so the report shows their label share too
    make_ep(tmp_path, 1)  # actions 0,1,2,0,1,2; the last step has no ammo
    s = dataset_stats([str(tmp_path)])
    assert s['action_share_valid'] == [0.4, 0.4, 0.2, 0.0, 0.0]


def test_plot_bc_curves_writes_png(tmp_path):
    rows = [dict(epoch=1, val_score=1.0, kills_per_shot=0.5),
            dict(epoch=2, val_score=2.0, kills_per_shot=0.6)]
    p = tmp_path / 'a.log.json'
    p.write_text(json.dumps(dict(config={}, log=rows)))
    plot_bc_curves({'a': str(p)}, str(tmp_path / 'c.png'))
    assert (tmp_path / 'c.png').exists()


from doomagent.reports import rounds_summary


def _log(d, name, score):
    p = d / f'{name}.pt.log.json'
    p.write_text(json.dumps(dict(config={}, log=[dict(epoch=1, val_score=score)])))
    return str(p)


def test_rounds_summary_stops_after_two_small_gains(tmp_path):
    logs = {n: _log(tmp_path, n, s) for n, s in (('r1', 10.0), ('r2', 10.3), ('r3', 10.6))}
    rows, stop, best = rounds_summary(logs)
    assert stop and best == 'r3' and rows[1]['gain'] == pytest.approx(0.3)


def test_rounds_summary_continues_after_a_real_gain(tmp_path):
    logs = {n: _log(tmp_path, n, s) for n, s in (('r1', 10.0), ('r2', 12.0))}
    assert rounds_summary(logs)[1] is False


from doomagent.reports import plot_ppo_curves


def test_plot_ppo_curves_writes_png(tmp_path):
    p = tmp_path / 'val_log.json'
    p.write_text(json.dumps([dict(timesteps=0, val_score=1.0), dict(timesteps=10, val_score=2.0)]))
    plot_ppo_curves({'ft': str(p)}, str(tmp_path / 'p.png'), bc_level=1.5)
    assert (tmp_path / 'p.png').exists()
