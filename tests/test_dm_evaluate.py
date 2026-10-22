import pytest

from dmfiles import FakeMatchEnv, row  # tests/ is on sys.path, as for epfiles
from doomagent.dm.evaluate import run_match, summarize
from doomagent.evaluate import val_score


def forward(obs, priv):
    return 0


def test_counts_suicides_kills_and_kd():
    rows = [row(frags=f, deaths=d, first=i in (0, 3)) for i, (f, d) in
            enumerate([(0, 0), (0, 0), (-1, 1), (-1, 1), (0, 1), (1, 1)])]
    r = run_match(FakeMatchEnv(rows), forward, 1)
    assert (r['frags'], r['deaths'], r['suicides'], r['kills'], r['kd']) == (1, 1, 1, 2, 1.0)


def test_shots_ignore_the_refill_after_respawn():
    rows = [row(ammo=a, first=i in (0, 4)) for i, a in enumerate([15, 14, 13, 13, 15, 14])]
    assert run_match(FakeMatchEnv(rows), forward, 1)['shots'] == 3


def test_stuck_share_counts_still_bot_free_steps():
    r = run_match(FakeMatchEnv([row(first=i == 0) for i in range(30)]), forward, 1)
    assert r['stuck_steps'] == 13 and r['stuck_share'] == pytest.approx(13 / 30)


def test_enemy_share_and_bot_median():
    rows = [row(bot=i < 10, first=i == 0) for i in range(30)]
    r = run_match(FakeMatchEnv(rows, player_frags=(0, 3, -1, 2, 0, 1, 1, 4)), forward, 1)
    assert r['enemy_share'] == pytest.approx(10 / 30) and r['bot_median'] == 1.0


def test_reward_is_frags_so_val_score_works():
    res = [dict(run_match(FakeMatchEnv([row(frags=f, first=True)]), forward, 1), variant='base')
           for f in (2, 4)]
    assert val_score(res)[0] == 3.0 and summarize(res)['frags']['mean'] == 3.0


def test_chooser_reset_once_per_match():
    class C:
        resets = 0

        def reset(self):
            self.resets += 1

        def __call__(self, obs, priv):
            return 0

    c = C()
    run_match(FakeMatchEnv([row(first=i == 0) for i in range(5)]), c, 1)
    assert c.resets == 1


def test_trace_has_one_row_per_step():
    r = run_match(FakeMatchEnv([row(first=i == 0) for i in range(7)]), forward, 1, trace=True)
    assert all(len(v) == 7 for v in r['trace'].values())


@pytest.mark.vizdoom
def test_parallel_matches_serial():
    from dataclasses import replace
    from doomagent.dm.config import DM
    from doomagent.evaluate import evaluate_plan
    from doomagent.variants import BASE
    cfg = replace(DM, timelimit=0.1)
    plan = [(BASE, (35011, 35012))]
    assert evaluate_plan(plan, forward, cfg) == evaluate_plan(plan, forward, cfg, workers=2)


def test_random_chooser_repeats_per_match():
    from doomagent.dm.evaluate import RandomChooser
    c = RandomChooser()
    runs = []
    for _ in range(2):
        c.reset()
        runs.append([run_match(FakeMatchEnv([row(first=i == 0) for i in range(20)]), c, 1)['length'],
                     c.actions[:]])
    assert runs[0] == runs[1] and len(set(runs[0][1])) > 3
