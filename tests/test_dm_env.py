import math
import os
from dataclasses import replace

import numpy as np
import pytest

from doomagent.dm.config import BOT, DM, FORWARD, N_ACTIONS, SHOOT, TURN_LEFT

pytestmark = pytest.mark.vizdoom
SHORT = replace(DM, timelimit=0.1)  # 210 tics, 53 steps


@pytest.fixture(scope='module')
def env():
    from doomagent.dm.env import DeathmatchEnv
    e = DeathmatchEnv(SHORT)
    yield e
    e.close()


def test_reset_observation(env):
    o = env.reset(35000)
    assert o.frames.shape == (4, 120, 160) and o.frames.dtype == np.uint8 and o.first
    assert all((o.frames[0] == o.frames[i]).all() for i in range(1, 4))
    assert o.vec.dtype == np.float32 and np.allclose(o.vec, [1.0, 0.3])  # 100 health, 15 rockets


def test_network_frame_is_the_downscaled_render(env):
    from doomagent.env import to_net
    o = env.reset(35001)
    assert (o.frames[-1] == to_net(env.screen(), SHORT)).all()
    o, _, _ = env.step(TURN_LEFT)
    assert (o.frames[-1] == to_net(env.screen(), SHORT)).all() and not o.first


def test_bots_join_by_name(env):
    env.reset(35002)
    env.step(FORWARD)
    p = env.privileged()
    assert p.players == SHORT.bots + 1 and len(p.player_frags) == SHORT.bots + 1


def test_match_repeats_bit_for_bit():
    from doomagent.dm.env import DeathmatchEnv
    acts = np.random.default_rng(0).integers(0, N_ACTIONS, 60)

    def run():
        e = DeathmatchEnv(SHORT)
        frames, rewards, done, i = [e.reset(35003).frames], [], False, 0
        while not done:
            o, r, done = e.step(int(acts[i]))
            frames.append(o.frames)
            rewards.append(r)
            i += 1
        e.close()
        return frames, rewards

    (fa, ra), (fb, rb) = run(), run()
    assert len(fa) == len(fb) and ra == rb
    assert all(np.array_equal(x, y) for x, y in zip(fa, fb))


def test_different_seeds_differ(env):
    a = [env.reset(35004).frames] + [env.step(FORWARD)[0].frames for _ in range(20)]
    b = [env.reset(35005).frames] + [env.step(FORWARD)[0].frames for _ in range(20)]
    assert not all(np.array_equal(x, y) for x, y in zip(a, b))


def test_neighbour_seeds_play_different_matches(env):
    # half of all seeds used to play one and the same match (see doomagent.vzseed)
    def bots(seed):
        env.reset(seed)
        for _ in range(3):
            env.step(FORWARD)
        return sorted((round(o.x), round(o.y)) for o in env.privileged().objects if o.name == BOT)

    assert bots(35004) != bots(35005)


def test_match_ends_at_the_time_limit(env):
    env.reset(35006)
    n, done = 0, False
    while not done:
        _, _, done = env.step(FORWARD)
        n += 1
    assert 50 <= n <= 53


def test_kill_respawns_with_a_fresh_stack(env):
    env.reset(35007)
    for _ in range(5):
        env.step(FORWARD)
    env.game.send_game_command('kill')
    o, r, done = env.step(FORWARD)
    p = env.privileged()
    assert o.first and r == -1.0 and not done
    assert p.deaths == 1 and p.frags == -1 and p.health > 0
    assert all((o.frames[0] == o.frames[i]).all() for i in range(1, 4))
    assert not env.step(FORWARD)[0].first


def test_own_player_is_not_in_labels_or_objects(env):
    env.reset(35008)
    for _ in range(40):
        p = env.privileged()
        for x, y, name in ([(lb.object_position_x, lb.object_position_y, lb.object_name) for lb in p.labels]
                           + [(o.x, o.y, o.name) for o in p.objects]):
            assert name != BOT or math.hypot(x - p.px, y - p.py) >= 1.0
        env.step(TURN_LEFT)


def test_rockets_are_counted(env):
    # the launcher fires on a fresh press only: holding SHOOT fires nothing, so presses alternate;
    # a rocket into a near wall may respawn the player with 15 again, hence the minimum
    env.reset(35009)
    seen = []
    for i in range(16):
        env.step(SHOOT if i % 2 == 0 else FORWARD)
        seen.append(env.privileged().ammo)
    assert min(seen) < 15


def test_held_shoot_fires_once_at_most(env):
    env.reset(35009)
    for _ in range(16):
        env.step(SHOOT)
    assert env.privileged().ammo >= 14


def test_sectors_are_static(env):
    env.reset(35000)
    a = env.sectors
    env.reset(35001)
    assert env.sectors == a and len(a) == 24


def test_tic_by_tic_step_matches_the_frame_skip_step():
    from doomagent.dm.env import DeathmatchEnv
    out = []
    for tics in (False, True):
        e = DeathmatchEnv(SHORT)
        o, rows = e.reset(35010), []
        for i in range(40):
            o, r, d = e.step(i % N_ACTIONS, on_tic=(lambda rgb, p: None) if tics else None)
            rows.append((o.frames.copy(), r, d))
        e.close()
        out.append(rows)
    assert all(np.array_equal(a[0], b[0]) and a[1:] == b[1:] for a, b in zip(*out))


@pytest.mark.skipif(os.name != 'nt', reason='Windows power throttling only')
def test_env_unthrottles_its_vizdoom_process(env):
    env.reset(35000)
    assert env.unthrottled
