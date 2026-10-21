import json

from doomagent.compare import budget, parse_pairs


def test_budget_sums_data_and_rl_steps(tmp_path):
    for name, steps in (('a', 100), ('b', 50)):
        d = tmp_path / name
        d.mkdir()
        (d / 'meta.json').write_text(json.dumps(dict(episodes=1, steps=steps)))
    got = budget([str(tmp_path / 'a'), str(tmp_path / 'b')], 200)
    assert got == dict(data_steps=150, rl_steps=200, total_steps=350)


def test_parse_pairs():
    assert parse_pairs(['bc0=runs/bc0.pt']) == {'bc0': 'runs/bc0.pt'}
    assert parse_pairs(None) == {}


def test_chooser_for_student_is_picklable_and_acts_like_it(tmp_path):
    # the test report runs episodes in worker processes; a student becomes a StudentChooser
    import pickle

    import numpy as np
    import torch

    from doomagent.compare import chooser_for
    from doomagent.model import StudentCNN
    torch.manual_seed(0)
    m = StudentCNN()
    torch.save(m.state_dict(), tmp_path / 'm.pt')
    c = pickle.loads(pickle.dumps(chooser_for(str(tmp_path / 'm.pt'))))
    obs = np.random.default_rng(0).integers(0, 255, (4, 120, 160), dtype=np.uint8)
    assert c(obs, None) == m.act(obs)
