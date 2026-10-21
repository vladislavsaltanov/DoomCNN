import json

import pytest

pytestmark = pytest.mark.vizdoom


def test_train_smoke_saves_best_checkpoint(tmp_path):
    from doomagent.env import DoomEnv
    from doomagent.record import episode_prefix, record_episode
    from doomagent.teacher import Teacher
    from doomagent.train_bc import train
    from doomagent.variants import BASE
    env = DoomEnv()
    for s in (1, 2):
        record_episode(env, s, lambda o, p, label: label, Teacher(),
                       episode_prefix(str(tmp_path), BASE, s))
    env.close()
    out = tmp_path / 'best.pt'
    train([str(tmp_path)], str(out), epochs=1, aug=True, eval_every=1, val=[(BASE, (5000,))])
    assert out.exists()
    log = json.loads((tmp_path / 'best.pt.log.json').read_text())
    assert log['log'][0]['epoch'] == 1 and 'val_score' in log['log'][0]
    # live progress for the dashboard: batches of the running epoch and the recent loss
    prog = json.loads((tmp_path / 'best.pt.progress.json').read_text())
    assert prog['epoch'] == 1 and prog['batch'] == prog['n_batches'] > 0
    assert prog['points'] and {'batch', 'loss', 'ce'} <= set(prog['points'][-1])


def test_progress_skips_an_update_when_the_file_is_busy(tmp_path, monkeypatch):
    # Windows refuses os.replace while the dashboard has the file open; training must go on
    import os
    from doomagent.train_bc import Progress
    p = Progress(str(tmp_path / 'x.progress.json'), n_batches=2, every=1)

    def busy(src, dst):
        raise PermissionError(5, 'Access is denied')
    monkeypatch.setattr(os, 'replace', busy)
    p.add(1, 1, dict(loss=1.0))  # must not raise
    monkeypatch.undo()
    p.add(1, 2, dict(loss=0.5))
    assert json.loads((tmp_path / 'x.progress.json').read_text())['batch'] == 2
