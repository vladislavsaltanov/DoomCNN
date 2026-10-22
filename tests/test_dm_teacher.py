import math
from dataclasses import replace

import numpy as np
import pytest

from doomagent.dm.config import (DM, FINE_LEFT, SHOOT, STRAFE_LEFT, STRAFE_RIGHT, TURN_LEFT)
from doomagent.dm.env import DMObs, DMPrivileged, MapObject
from doomagent.dm.teacher import DMTeacher
from doomagent.env import Label


def sector(points, floor=0.0, ceil=128.0):
    n = len(points)
    return (floor, ceil, tuple((*points[i], *points[(i + 1) % n], True) for i in range(n)))


# a 1024 room and a sealed room nobody can walk to
MAP = [sector([(0, 0), (1024, 0), (1024, 1024), (0, 1024)]),
       sector([(2000, 2000), (2128, 2000), (2128, 2128), (2000, 2128)])]
ME = (256.0, 512.0)


def place(rel, dist, me=ME, angle=0.0):
    # world position and network-pixel box of something `dist` away, `rel` degrees left of the view
    a = math.radians(angle + rel)
    wx, wy = me[0] + dist * math.cos(a), me[1] + dist * math.sin(a)
    xn, half = -math.tan(math.radians(rel)), 16.0 / max(dist, 1.0)
    return wx, wy, (xn - half + 1) * 80, 2 * half * 80


def bot(id, rel, dist, me=ME, north=0.0, pos=None):
    wx, wy, x, w = place(rel, dist, me)
    if pos is not None:
        wx, wy = pos
    return Label(1, id, 'DoomPlayer', x, 40.0, w, 40.0, wx, wy + north)


def item(name, rel, dist, me=ME):
    wx, wy, x, w = place(rel, dist, me)
    return Label(2, 500, name, x, 60.0, w, 20.0, wx, wy)


def priv(**kw):
    d = dict(px=ME[0], py=ME[1], angle=0.0, health=100.0, ammo=10.0, attack_ready=True, tic=100,
             sectors=MAP)
    d.update(kw)
    return DMPrivileged(**d)


def obs(first=False):
    return DMObs(np.zeros((4, 120, 160), np.uint8), np.zeros(2, np.float32), first)


def scripted_steps():
    # a bot appears, moves north and leaves; a medikit passes by
    out = []
    for t in range(20):
        labels = []
        if 3 <= t < 9:
            labels.append(bot(4, 0.0, 500, north=8.0 * (t - 3)))
        if 10 <= t < 13:
            labels.append(item('Medikit', 20.0, 300))
        out.append((obs(first=t == 0), priv(labels=labels, tic=100 + 4 * t, health=40.0)))
    return out


def test_shoots_a_still_bot_in_the_crosshair():
    assert DMTeacher(DM)(obs(), priv(labels=[bot(1, 0.0, 400)])) == SHOOT


def test_no_shot_inside_safe_distance():
    assert DMTeacher(DM)(obs(), priv(labels=[bot(1, 0.0, 100)])) in (STRAFE_LEFT, STRAFE_RIGHT)


def test_no_shot_while_reloading():
    assert DMTeacher(DM)(obs(), priv(labels=[bot(1, 0.0, 400)], attack_ready=False)) != SHOOT


def test_no_shot_when_a_wall_is_close():
    assert DMTeacher(DM)(obs(), priv(px=900.0, labels=[bot(1, 0.0, 400, me=(900.0, 512.0))])) != SHOOT


def test_no_shot_through_a_close_barrel():
    labels = [bot(1, 0.0, 400), item('ExplosiveBarrel', 0.0, 120)]
    assert DMTeacher(DM)(obs(), priv(labels=labels)) != SHOOT


def test_leads_a_moving_bot():
    t = DMTeacher(DM)
    t(obs(), priv(labels=[bot(1, 0.0, 600)], tic=100))
    a = t(obs(), priv(labels=[bot(1, 0.0, 600, north=16.0)], tic=104))  # 4 units per tic north
    assert t.lead == pytest.approx(-0.2, abs=0.05) and a in (TURN_LEFT, FINE_LEFT)


def test_fresh_track_has_no_lead():
    t = DMTeacher(DM)
    t(obs(), priv(labels=[bot(1, 0.0, 600)]))
    assert t.lead == 0.0


def test_ignores_corpses_and_itself():
    t = DMTeacher(DM)
    t(obs(), priv(labels=[item('DeadDoomPlayer', 0.0, 300), bot(1, 0.0, 400, pos=ME)]))
    assert t.mode != 'combat'


def test_items_only_after_seen():
    t = DMTeacher(DM)
    t(obs(), priv(health=30.0))
    assert t.mode == 'patrol'
    t(obs(), priv(health=30.0, labels=[item('Medikit', 0.0, 344)]))  # at (600, 512)
    t(obs(), priv(health=30.0))
    assert t.mode == 'items' and math.hypot(t.goal[0] - 600, t.goal[1] - 512) < 1


def test_teacher_ignores_objects():
    seq = scripted_steps()
    full = [MapObject(i, n, 300.0 + 40 * i, 300.0, 0.0) for i, n in enumerate(['DoomPlayer', 'Medikit'] * 4)]
    a, b = DMTeacher(DM), DMTeacher(DM)
    out_a = [(a(o, replace(p, objects=[])), a.mode) for o, p in seq]
    out_b = [(b(o, replace(p, objects=full)), b.mode) for o, p in seq]
    assert out_a == out_b


def test_known_item_forgotten_when_reached():
    t = DMTeacher(DM)
    t(obs(), priv(labels=[item('Medikit', 0.0, 344)]))
    t(obs(), priv(px=590.0))
    assert not t.known


def test_chase_goes_to_last_seen_then_gives_up():
    t = DMTeacher(DM)
    t(obs(), priv(labels=[bot(1, 0.0, 400)], tic=100))
    t(obs(), priv(tic=104))
    assert t.mode == 'chase' and math.hypot(t.goal[0] - 656, t.goal[1] - 512) < 1
    t(obs(), priv(tic=100 + 2 * 35 + 4))
    assert t.mode == 'patrol'


def test_no_ammo_means_items_not_combat():
    t = DMTeacher(DM)
    t(obs(), priv(labels=[item('RocketAmmo', 30.0, 300)]))
    t(obs(), priv(ammo=0.0, labels=[bot(1, 0.0, 400)]))
    assert t.mode == 'items'


def test_respawn_clears_chase_and_lock_keeps_items():
    t = DMTeacher(DM)
    t(obs(), priv(labels=[bot(1, 0.0, 400), item('Medikit', 30.0, 300)]))
    t(obs(first=True), priv(px=100.0, py=100.0))
    assert t.last_seen is None and t.target is None and len(t.known) == 1


def test_unreachable_goal_falls_back_to_patrol():
    t = DMTeacher(DM)
    t.reset()
    t(obs(), priv())
    t.known.append(('health', 2064.0, 2064.0))  # in the sealed sector: no path
    t(obs(), priv(health=30.0))
    assert t.mode == 'patrol' and not t.known


@pytest.mark.vizdoom
def test_teacher_patrols_cig_without_getting_stuck():
    from doomagent.dm.env import DeathmatchEnv
    from doomagent.dm.evaluate import run_match
    e = DeathmatchEnv(replace(DM, bots=0, timelimit=0.5))
    t = DMTeacher(DM)
    seen = []
    r = run_match(e, lambda o, p: (t(o, p), seen.append(t.ring_i))[0], 35000)
    e.close()
    assert r['stuck_share'] < 0.05 and r['path'] > 1500 and len(set(seen)) >= 3
