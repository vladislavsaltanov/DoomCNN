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
