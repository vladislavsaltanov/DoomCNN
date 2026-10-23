import pytest

from dmfiles import FakeMatchEnv, row  # tests/dm is on sys.path
from doomagent.dm.actions import HEAD_SIZES, NOOP
from doomagent.dm.evaluate import run_match, summarize
from doomagent.evaluate import val_score


def forward(obs, priv):
    return NOOP


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
            return NOOP

    c = C()
    run_match(FakeMatchEnv([row(first=i == 0) for i in range(5)]), c, 1)
    assert c.resets == 1


def test_trace_has_one_row_per_step():
    r = run_match(FakeMatchEnv([row(first=i == 0) for i in range(7)]), forward, 1, trace=True)
    assert all(len(v) == 7 for v in r['trace'].values()) and r['trace']['action'].shape == (7, 4)


@pytest.mark.vizdoom
def test_parallel_matches_serial():
    from dataclasses import replace
    from doomagent.dm.config import DM
    from doomagent.dm.evaluate import evaluate_plan
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
    acts = runs[0][1]
    assert all(len(a) == 4 and all(0 <= v < k for v, k in zip(a, HEAD_SIZES)) for a in acts)
    assert len({a[0] for a in acts}) > 3 and len({a[1] for a in acts}) > 3


def test_shots_ignore_weapon_switch():
    # pistol 50 -> shotgun 20 is a switch, 20 -> 19 is a shot
    rows = [row(ammo=a, weapon=w, first=i == 0) for i, (a, w) in enumerate([(50, 2), (20, 3), (19, 3), (19, 3)])]
    assert run_match(FakeMatchEnv(rows), forward, 1)['shots'] == 1


def test_weapon_share():
    rows = [row(weapon=w, first=i == 0) for i, w in enumerate([2, 2, 3, 5])]
    r = run_match(FakeMatchEnv(rows), forward, 1)
    assert r['weapon_share'][2] == 0.5 and r['weapon_share'][5] == 0.25


def test_items_count_pickups_not_respawn_refills():
    # ITEMCOUNT ignores medikits and rockets: a pickup is a rise in health or rockets in one life
    rows = [row(ammo=a, health=h, first=i in (0, 3)) for i, (a, h) in
            enumerate([(10, 60), (15, 60), (15, 85), (15, 100), (15, 100)])]
    assert run_match(FakeMatchEnv(rows), forward, 1)['items'] == 2


def test_summarize_weapon_share():
    res = [run_match(FakeMatchEnv([row(weapon=w, first=i == 0) for i, w in enumerate(ws)]), forward, 1)
           for ws in ([2, 2], [3, 3])]
    sh = summarize(res)['weapon_share']
    assert sh[2] == 0.5 and sh[3] == 0.5 and sh[5] == 0.0


def test_rank_among_players():
    rows = [row(frags=3, first=True)]
    r = run_match(FakeMatchEnv(rows, player_frags=(3, 5, 3, 1, 0, 0, 0, 0)), forward, 1)
    assert r['rank'] == 2   # one bot has more, a tie does not push the player down
    res = [r, run_match(FakeMatchEnv(rows, player_frags=(3, 2, 1, 0, 0, 0, 0, 0)), forward, 1)]
    assert summarize(res)['first_share'] == 0.5


def test_live_tap_shows_a_student_match(tmp_path):
    # the network's val matches are shown live too: frame, state and the finished match
    import json
    from doomagent.dm.livetap import LiveTap
    tap = LiveTap(str(tmp_path), every_s=0.0, worker='e1')
    rows = [row(frags=float(i // 4), first=i == 0) for i in range(8)]
    r = run_match(FakeMatchEnv(rows, player_frags=(0, 3, 1, 0, 0, 0, 0, 0)), forward, 9, tap=tap)
    st = json.load(open(tmp_path / 'e1.json'))
    assert st['seed'] == 9 and st['mode'] == 'net' and st['label'] == st['action']
    assert (tmp_path / 'e1.screen.jpg').stat().st_size > 0
    done = json.load(open(tmp_path / 'e1.done.json'))
    assert done[0]['frags'] == r['frags'] and done[0]['best_bot'] == 3.0 and done[0]['first'] is False
