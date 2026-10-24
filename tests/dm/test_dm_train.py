import json
import os
from dataclasses import replace

import numpy as np
import pytest

from dmfiles import make_match
from doomagent.dm.actions import HEAD_SIZES, HEADS
from doomagent.dm.config import DM
from doomagent.dm.train import train


@pytest.mark.parametrize('kind', ['m0', 'm1'])
def test_train_smoke_writes_outputs(tmp_path, kind):
    for s in range(3):
        make_match(tmp_path / 'd', s, T=70, first_at=(0, 30))
    out = str(tmp_path / f'{kind}.pt')
    train(kind, [str(tmp_path / 'd')], out, epochs=1, val=[], workers=0, batch_steps=64)
    log = json.load(open(out + '.log.json'))
    live = json.load(open(out + '.live.json'))
    assert os.path.exists(out) and log['log'] and list(live['confusion']) == list(HEADS)
    assert all(np.array(live['confusion'][h]).shape == (k, k) for h, k in zip(HEADS, HEAD_SIZES))
    assert all(f'holdout_acc_{h}' in log['log'][-1] for h in HEADS)
    assert log['config']['holdout_files']
    assert not set(log['config']['holdout_files']) & set(log['config']['train_files'])
    assert log['log'][-1]['lr'] < 1e-6  # cosine schedule ends at zero
    assert os.path.exists(out + '.progress.json')


def test_live_json_skips_when_busy(tmp_path, monkeypatch):
    from doomagent.dm import train as T

    def busy(*a, **k):
        raise PermissionError('the dashboard is reading')

    monkeypatch.setattr(T.os, 'replace', busy)
    T.write_json_atomic(str(tmp_path / 'x.json'), {'a': 1})  # must not raise


@pytest.mark.vizdoom
def test_train_with_val_on_a_short_match(tmp_path):
    from doomagent.variants import BASE
    for s in range(2):
        make_match(tmp_path / 'd', s, T=60)
    out = str(tmp_path / 'm1.pt')
    train('m1', [str(tmp_path / 'd')], out, epochs=1, val=[(BASE, (35000,))], workers=0,
          eval_workers=1, batch_steps=64, cfg=replace(DM, timelimit=0.1))
    assert 'val_score' in json.load(open(out + '.log.json'))['log'][-1]


def test_progress_knows_the_epochs_and_the_start(tmp_path):
    make_match(tmp_path / 'd', 0, T=70, first_at=(0,))
    make_match(tmp_path / 'd', 1, T=70, first_at=(0,))
    out = str(tmp_path / 'm0.pt')
    train('m0', [str(tmp_path / 'd')], out, epochs=2, val=[], workers=0, batch_steps=64)
    p = json.load(open(out + '.progress.json'))
    assert p['epochs'] == 2 and p['start'] <= p['t']


@pytest.mark.parametrize('kind', ['m0', 'm1'])
def test_train_with_aux_heads(tmp_path, kind):
    for s in range(3):
        make_match(tmp_path / 'd', s, T=70, first_at=(0, 30), aux=True)
    out = str(tmp_path / f'{kind}.pt')
    train(kind, [str(tmp_path / 'd')], out, epochs=1, val=[], workers=0, batch_steps=64, aux_w=1.0)
    row = json.load(open(out + '.log.json'))['log'][-1]
    assert {'holdout_aux_pos_acc', 'holdout_aux_x_acc', 'holdout_aux_enemy_acc', 'train_aux'} <= set(row)
    from doomagent.dm.model import load_model
    assert load_model(out).aux_on


@pytest.mark.parametrize('kind', ['m0', 'm1'])
def test_train_with_spatial_heads(tmp_path, kind):
    for s in range(3):
        make_match(tmp_path / 'd', s, T=70, first_at=(0, 30), aux=True, spatial=True)
    out = str(tmp_path / f'{kind}.pt')
    train(kind, [str(tmp_path / 'd')], out, epochs=1, val=[], workers=0, batch_steps=64, aux_w=1.0, spatial_w=1.0)
    row = json.load(open(out + '.log.json'))['log'][-1]
    assert {'holdout_sp_map_acc', 'holdout_sp_odo_err', 'holdout_sp_fut_err', 'holdout_sp_goal_dir_acc',
            'holdout_aux_pos_acc', 'train_spatial'} <= set(row)
    from doomagent.dm.model import load_model
    m = load_model(out)
    assert m.spatial_on and m.aux_on


def test_train_with_folder_weights(tmp_path):
    for s in range(3):
        make_match(tmp_path / 'd', s, T=70, first_at=(0, 30))
    make_match(tmp_path / 'e', 9, T=70, first_at=(0,))
    out = str(tmp_path / 'm1.pt')
    train('m1', [str(tmp_path / 'd'), str(tmp_path / 'e')], out, epochs=1, val=[], workers=0, batch_steps=64,
          dir_weights={str(tmp_path / 'e'): 4.0})
    assert json.load(open(out + '.log.json'))['config']['dir_weights'] == {str(tmp_path / 'e'): 4.0}
