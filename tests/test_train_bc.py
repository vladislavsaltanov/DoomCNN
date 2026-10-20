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
