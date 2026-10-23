import math
import os
from dataclasses import replace

import numpy as np
import pytest

from doomagent.dm.actions import FORWARD, HEAD_SIZES, NOOP, STAY, TURN_DEG, make_vec, turn_index
from doomagent.dm.config import BOT, DM

pytestmark = pytest.mark.vizdoom
SHORT = replace(DM, timelimit=0.1)  # 210 tics, 53 steps
FWD = (FORWARD, 0, 0, 0)
LEFT10 = (STAY, turn_index(10.0), 0, 0)


@pytest.fixture(scope='module')
def env():
    from doomagent.dm.env import DeathmatchEnv
    e = DeathmatchEnv(SHORT)
    yield e
    e.close()


def wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def test_reset_observation(env):
    o = env.reset(35000)
    assert o.frames.shape == (4, SHORT.height, SHORT.width) and o.frames.dtype == np.uint8 and o.first
    assert all((o.frames[0] == o.frames[i]).all() for i in range(1, 4))
    assert o.vec.dtype == np.float32 and np.allclose(o.vec, make_vec(100.0, 0.0, 2, 50.0))


def test_start_weapon_is_pistol(env):
    o = env.reset(35000)
    p = env.privileged()
    assert p.weapon == 2 and p.weapon_ammo == 50 and p.ammo == 50 and o.vec[3:].argmax() == 1
    assert p.owned[1] == 1 and p.owned[2] == 1 and len(p.owned) == len(p.slot_ammo) == 8


def test_network_frame_is_the_downscaled_render(env):
    from doomagent.env import to_net
    o = env.reset(35001)
    assert (o.frames[-1] == to_net(env.screen(), SHORT)).all()
    o, _, _ = env.step(LEFT10)
    assert (o.frames[-1] == to_net(env.screen(), SHORT)).all() and not o.first


@pytest.mark.parametrize('i', range(1, len(TURN_DEG)))
def test_turn_is_exact(env, i):
    env.reset(35011)
    for _ in range(6):  # a player who just teleported in cannot move for a few tics
        env.step(NOOP)
    a0 = env.privileged().angle
    env.step((STAY, i, 0, 0))
    assert abs(wrap(env.privileged().angle - a0) - TURN_DEG[i]) <= 0.2


def test_next_weapon_once_per_step(env):
    # pistol -> the next owned weapon (the fist); waiting steps never switch again, a second press goes back
    env.reset(35012)
    env.step((STAY, 0, 0, 1))
    seen = [env.privileged().weapon]
    for _ in range(6):
        env.step(NOOP)
        seen.append(env.privileged().weapon)
    assert seen[-1] == 1 and len(set(seen[seen.index(1):])) == 1
    env.step((STAY, 0, 0, 1))
    for _ in range(6):
        env.step(NOOP)
    assert env.privileged().weapon == 2


def test_pistol_shots_are_counted(env):
    env.reset(35009)
    for _ in range(6):
        env.step((STAY, 0, 1, 0))
    assert env.privileged().weapon_ammo < 50


def test_bots_join_by_name(env):
    env.reset(35002)
    env.step(FWD)
    p = env.privileged()
    assert p.players == SHORT.bots + 1 and len(p.player_frags) == SHORT.bots + 1


def random_actions(n, seed=0):
    g = np.random.default_rng(seed)
    return [tuple(int(g.integers(0, k)) for k in HEAD_SIZES) for _ in range(n)]


def test_match_repeats_bit_for_bit():
    from doomagent.dm.env import DeathmatchEnv
    acts = random_actions(60)

    def run():
        e = DeathmatchEnv(SHORT)
        frames, rewards, done, i = [e.reset(35003).frames], [], False, 0
        while not done:
            o, r, done = e.step(acts[i])
            frames.append(o.frames)
            rewards.append(r)
            i += 1
        e.close()
        return frames, rewards

    (fa, ra), (fb, rb) = run(), run()
    assert len(fa) == len(fb) and ra == rb
    assert all(np.array_equal(x, y) for x, y in zip(fa, fb))


def test_neighbour_seeds_play_different_matches(env):
    def bots(seed):
        env.reset(seed)
        for _ in range(3):
            env.step(FWD)
        return sorted((round(o.x), round(o.y)) for o in env.privileged().objects if o.name == BOT)

    assert bots(35004) != bots(35005)


def test_match_ends_at_the_time_limit(env):
    env.reset(35006)
    n, done = 0, False
    while not done:
        _, _, done = env.step(FWD)
        n += 1
    assert 50 <= n <= 53


def test_kill_respawns_with_a_fresh_stack(env):
    env.reset(35007)
    for _ in range(5):
        env.step(FWD)
    env.game.send_game_command('kill')
    o, r, done = env.step(FWD)
    p = env.privileged()
    assert o.first and r == -1.0 and not done
    assert p.deaths == 1 and p.frags == -1 and p.health > 0 and p.weapon == 2
    assert all((o.frames[0] == o.frames[i]).all() for i in range(1, 4))
    assert not env.step(FWD)[0].first


def test_own_player_is_not_in_labels_or_objects(env):
    env.reset(35008)
    for _ in range(40):
        p = env.privileged()
        for x, y, name in ([(lb.object_position_x, lb.object_position_y, lb.object_name) for lb in p.labels]
                           + [(o.x, o.y, o.name) for o in p.objects]):
            assert name != BOT or math.hypot(x - p.px, y - p.py) >= 1.0
        env.step(LEFT10)


def test_visible_pixels_of_every_label(env):
    # ViZDoom also labels objects hidden behind walls and ledges: their box is set, their pixels are 0
    env.reset(35009)
    seen = 0
    for _ in range(60):
        p = env.privileged()
        assert set(p.vis_px) == {int(lb.object_id) for lb in p.labels}
        assert all(v >= 0 for v in p.vis_px.values())
        seen += sum(p.vis_px[int(lb.object_id)] > 0 for lb in p.labels if lb.object_name == BOT)
        env.step(LEFT10)
    assert seen > 0


def test_sectors_are_static(env):
    env.reset(35000)
    a = env.sectors
    env.reset(35001)
    assert env.sectors == a and len(a) == 179


def test_tic_by_tic_step_matches_the_frame_skip_step():
    from doomagent.dm.env import DeathmatchEnv
    acts = random_actions(40, seed=1)
    out = []
    for tics in (False, True):
        e = DeathmatchEnv(SHORT)
        o, rows = e.reset(35010), []
        for a in acts:
            o, r, d = e.step(a, on_tic=(lambda rgb, p: None) if tics else None)
            rows.append((o.frames.copy(), r, d))
        e.close()
        out.append(rows)
    assert all(np.array_equal(a[0], b[0]) and a[1:] == b[1:] for a, b in zip(*out))


@pytest.mark.skipif(os.name != 'nt', reason='Windows power throttling only')
def test_env_unthrottles_its_vizdoom_process(env):
    env.reset(35000)
    assert env.unthrottled


def test_run_and_autoaim_are_set_on_every_game():
    # cl_run is an archived cvar: one game with +cl_run 1 writes it to _vizdoom.ini for every later game,
    # so each game sets it explicitly
    from doomagent.dm.env import game_args
    a = game_args(replace(DM, run=False, autoaim=False))
    assert '+cl_run 0' in a and '+sv_noautoaim 1' in a
    b = game_args(replace(DM, run=True, autoaim=True))
    assert '+cl_run 1' in b and '+sv_noautoaim 0' in b
    assert ' ' not in a.split('+viz_bots_path ')[1]
