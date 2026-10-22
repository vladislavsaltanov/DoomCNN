from doomagent import config as C
from doomagent.config import Cfg
from doomagent.dm import config as D
from doomagent.variants import BASE


def test_old_defaults_unchanged():
    c = Cfg()
    assert c.scenario == 'defend_the_center'
    assert (c.frame_skip, c.stack, c.height, c.width, c.turn_mode, c.fine_turn_delta, c.aim_margin,
            c.min_target_w, c.noise_p, c.lam, c.lam_seg) == (4, 4, 120, 160, 'binary', 0.5, 0.2, 0.0,
                                                             0.10, 0.5, 0.5)


def test_deathmatch_cfg():
    d = D.DM
    assert d.scenario == 'deathmatch'
    assert (d.bots, d.timelimit, d.safe_dist, d.chase_s, d.low_health, d.low_ammo, d.seg_len,
            d.burn_in) == (7, 2.0, 128.0, 1.0, 50.0, 5.0, 32, 8)  # D1 decisions
    assert d.aim_margin == 0.0
    assert D.N_ACTIONS == len(D.ACTION_NAMES) == 10


def test_mirror_swaps_left_and_right_including_strafe():
    for a, b in ((D.FORWARD_LEFT, D.FORWARD_RIGHT), (D.TURN_LEFT, D.TURN_RIGHT),
                 (D.FINE_LEFT, D.FINE_RIGHT), (D.STRAFE_LEFT, D.STRAFE_RIGHT)):
        assert D.MIRROR[a] == b and D.MIRROR[b] == a
    assert D.MIRROR[D.FORWARD] == D.FORWARD and D.MIRROR[D.SHOOT] == D.SHOOT
    assert sorted(D.MIRROR) == list(range(D.N_ACTIONS))


def test_noise_actions_are_every_move_and_turn():
    assert set(D.NOISE_ACTIONS) == set(range(D.N_ACTIONS)) - {D.SHOOT}


def test_seed_sets_disjoint_and_new():
    new = [D.TRAIN_SEEDS, D.VARIANT_SEEDS, D.DAGGER_SEEDS, D.VAL_SEEDS, D.FRESH_SEEDS,
           D.CHECK_SEEDS, D.PILOT_SEEDS, D.TEST_SEEDS, D.RL_SEEDS]
    flat = [s for ss in new for s in ss]
    assert len(flat) == len(set(flat))
    old = set(C.TRAIN_SEEDS + C.TRAIN_SEEDS_EXTRA + C.DAGGER_SEEDS + C.VARIANT_SEEDS + C.VAL_SEEDS
              + C.TEST_SEEDS + C.RL_SEEDS) | set(range(5000, 8000))
    assert not old & set(flat)
    assert (len(D.VAL_SEEDS), len(D.FRESH_SEEDS), len(D.TEST_SEEDS), len(D.PILOT_SEEDS)) == (20, 40, 40, 10)


def test_plans_without_variants(tmp_path):
    root = str(tmp_path)
    assert D.val_plan(root) == [(BASE, D.VAL_SEEDS)]
    assert D.fresh_plan(root) == [(BASE, D.FRESH_SEEDS)]
    assert D.test_plans(root) == {'id': [(BASE, D.TEST_SEEDS)], 'ood': []}
    seeds = [[s for _, ss in D.dagger_plan(r, root=root) for s in ss] for r in (1, 2, 3)]
    assert all(len(s) == 40 for s in seeds)
    assert not set(seeds[0]) & set(seeds[1]) and not set(seeds[1]) & set(seeds[2])


def test_bot_roster_follows_bots_cfg():
    assert D.BOT_NAMES[:7] == ('Rambo', 'McClane', 'MacGyver', 'Plissken', 'Machete', 'Anderson', 'Leone')
    assert len(D.BOT_NAMES) == 15
