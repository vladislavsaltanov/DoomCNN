import math
from dataclasses import replace

import numpy as np
import pytest

from dmfiles import Poison
from doomagent.dm.actions import (BACK, BACK_LEFT, BACK_RIGHT, FORWARD, FORWARD_LEFT, FORWARD_RIGHT, STRAFE_LEFT,
                                  STRAFE_RIGHT, TURN_DEG, VEC_DIM)
from doomagent.dm.config import DM
from doomagent.dm.env import DMObs, DMPrivileged, MapObject
from doomagent.dm.teacher import DMTeacher
from doomagent.env import Label

SIDE_MOVES = {STRAFE_LEFT, STRAFE_RIGHT, FORWARD_LEFT, FORWARD_RIGHT, BACK_LEFT, BACK_RIGHT}


def sector(points, floor=0.0, ceil=128.0):
    n = len(points)
    return (floor, ceil, tuple((*points[i], *points[(i + 1) % n], True) for i in range(n)))


# a 1024 room and a sealed room nobody can walk to
MAP = [sector([(0, 0), (1024, 0), (1024, 1024), (0, 1024)]),
       sector([(2000, 2000), (2128, 2000), (2128, 2128), (2000, 2128)])]
ME = (256.0, 512.0)
ROCKETS = dict(weapon=5, weapon_ammo=10.0, ammo=10.0, owned=(0, 1, 1, 0, 0, 1, 0, 0),
               slot_ammo=(0, 0, 50, 0, 50, 10, 0, 0))


def place(rel, dist, me=ME, angle=0.0):
    # world position and network-pixel box of something `dist` away, `rel` degrees left of the view
    a = math.radians(angle + rel)
    wx, wy = me[0] + dist * math.cos(a), me[1] + dist * math.sin(a)
    xn, half = -math.tan(math.radians(rel)), 16.0 / max(dist, 1.0)
    return wx, wy, (xn - half + 1) * DM.width / 2, half * DM.width


def bot(id, rel, dist, me=ME, north=0.0, pos=None):
    wx, wy, x, w = place(rel, dist, me)
    if pos is not None:
        wx, wy = pos
    return Label(1, id, 'DoomPlayer', x, 40.0, w, 40.0, wx, wy + north)


def thing(name, x, y, id=900):
    return Label(3, id, name, DM.width / 2, 60.0, 4.0, 4.0, x, y)


def priv(**kw):
    d = dict(px=ME[0], py=ME[1], angle=0.0, health=100.0, attack_ready=True, tic=100, sectors=MAP, **ROCKETS)
    d.update(kw)
    return DMPrivileged(**d)


def obs(first=False):
    return DMObs(np.zeros((DM.stack, DM.height, DM.width), np.uint8), np.zeros(VEC_DIM, np.float32), first)


def test_shoots_and_strafes_in_one_step():
    move, turn, fire, weapon = DMTeacher(DM)(obs(), priv(labels=[bot(1, 0.0, 600)]))
    assert fire == 1 and move in SIDE_MOVES and TURN_DEG[turn] == 0.0 and weapon == 0


def test_turns_towards_target_with_big_turn():
    _, turn, fire, _ = DMTeacher(DM)(obs(), priv(labels=[bot(1, 40.0, 600)]))
    assert TURN_DEG[turn] == 40.0 and fire == 0


def test_no_rocket_close_switches_to_shotgun():
    p = priv(labels=[bot(1, 0.0, 100)], owned=(0, 1, 1, 1, 0, 1, 0, 0), slot_ammo=(0, 0, 50, 20, 50, 10, 0, 0))
    _, _, fire, weapon = DMTeacher(DM)(obs(), p)
    assert fire == 0 and weapon == 1


def test_no_rocket_into_a_near_wall():
    me = (900.0, 512.0)  # the east wall is 124 units ahead
    _, _, fire, _ = DMTeacher(DM)(obs(), priv(px=me[0], py=me[1], labels=[bot(1, 0.0, 400, me=me)]))
    assert fire == 0


def test_rocket_needs_a_fresh_press():
    t = DMTeacher(DM)
    assert t(obs(), priv(labels=[bot(1, 0.0, 600)]))[2] == 1
    assert t(obs(), priv(labels=[bot(1, 0.0, 600)], tic=104))[2] == 0


def test_chaingun_holds_fire():
    t = DMTeacher(DM)
    p = dict(weapon=4, weapon_ammo=50.0, ammo=50.0, owned=(0, 1, 1, 0, 1, 0, 0, 0), slot_ammo=(0, 0, 50, 0, 50, 0, 0, 0))
    assert t(obs(), priv(labels=[bot(1, 0.0, 600)], **p))[2] == 1
    assert t(obs(), priv(labels=[bot(1, 0.0, 600)], tic=104, attack_ready=False, **p))[2] == 1


def test_leads_a_moving_bot_with_rockets():
    t = DMTeacher(DM)
    t(obs(), priv(labels=[bot(1, 0.0, 600)], tic=100))
    t(obs(), priv(labels=[bot(1, 0.0, 600, north=40.0)], tic=104))  # 10 units per tic north
    assert t.aim[1] > ME[1] + 40.0 + 100.0


def test_hitscan_aims_at_the_bot():
    t = DMTeacher(DM)
    p = dict(weapon=4, weapon_ammo=50.0, ammo=50.0, owned=(0, 1, 1, 0, 1, 0, 0, 0), slot_ammo=(0, 0, 50, 0, 50, 0, 0, 0))
    t(obs(), priv(labels=[bot(1, 0.0, 600)], tic=100, **p))
    t(obs(), priv(labels=[bot(1, 0.0, 600, north=40.0)], tic=104, **p))
    assert t.aim == pytest.approx((ME[0] + 600.0, ME[1] + 40.0))


def test_dodges_an_incoming_rocket():
    # an enemy rocket from the east, 20 units per tic, passing 8 units north of the player: step south (right)
    t = DMTeacher(DM)
    t(obs(), priv(labels=[thing('Rocket', 856.0, 520.0, id=77)], tic=100))
    move, _, _, _ = t(obs(), priv(labels=[thing('Rocket', 776.0, 520.0, id=77)], tic=104))
    assert move == STRAFE_RIGHT


def test_teacher_does_not_cycle_weapons():
    t = DMTeacher(DM)
    held = [t(obs(), priv(labels=[bot(1, 0.0, 600)], tic=100 + 4 * i))[3] for i in range(20)]
    assert sum(held) == 0  # the rocket launcher is the best at 600
    t = DMTeacher(DM)
    p = dict(weapon=2, weapon_ammo=50.0, ammo=50.0)  # the pistol in hand, the rocket launcher owned
    presses = [t(obs(), priv(labels=[bot(1, 0.0, 600)], tic=100 + 4 * i, **p))[3] for i in range(20)]
    assert 1 <= sum(presses) <= 7 and all(not (a and b) for a, b in zip(presses, presses[1:]))


def test_bot_outside_labels_is_ignored():
    steps = [(obs(first=i == 0), priv(tic=100 + 4 * i, py=512.0 + 8 * i)) for i in range(12)]
    ghost = [MapObject(7, 'DoomPlayer', 700.0, 700.0, 0.0)]
    a, b = DMTeacher(DM), DMTeacher(DM)
    out_a = [(a(o, p), a.mode) for o, p in steps]
    out_b = [(b(o, replace(p, objects=ghost)), b.mode) for o, p in steps]
    assert out_a == out_b and all(m != 'combat' for _, m in out_b)


def test_objects_read_only_on_first_step():
    t = DMTeacher(DM)
    t(obs(first=True), priv(objects=[MapObject(1, 'Medikit', 600.0, 512.0, 0.0)]))
    for i in range(1, 6):
        t(obs(), priv(objects=Poison(), tic=100 + 4 * i))


def test_gear_goes_to_nearest_weapon():
    t = DMTeacher(DM)
    p = priv(weapon=2, weapon_ammo=50.0, ammo=50.0, owned=(0, 1, 1, 0, 0, 0, 0, 0), slot_ammo=(0, 0, 50, 0, 50, 0, 0, 0),
             objects=[MapObject(1, 'Chaingun', 600.0, 512.0, 0.0), MapObject(2, 'Chaingun', 900.0, 900.0, 0.0)])
    t(obs(first=True), p)
    assert t.mode == 'gear' and math.hypot(t.goal[0] - 600, t.goal[1] - 512) < 1


def test_low_health_goes_to_a_medikit():
    t = DMTeacher(DM)
    t(obs(first=True), priv(health=30.0, objects=[MapObject(1, 'Medikit', 600.0, 700.0, 0.0)]))
    assert t.mode == 'items' and math.hypot(t.goal[0] - 600, t.goal[1] - 700) < 1


def test_ignores_corpses_and_itself():
    t = DMTeacher(DM)
    t(obs(), priv(labels=[thing('DeadDoomPlayer', 500.0, 512.0), bot(1, 0.0, 400, pos=ME)]))
    assert t.mode != 'combat'


def test_chase_goes_to_last_seen_then_gives_up():
    t = DMTeacher(DM)
    t(obs(), priv(labels=[bot(1, 0.0, 400)], tic=100))
    t(obs(), priv(tic=104))
    assert t.mode == 'chase' and math.hypot(t.goal[0] - 656, t.goal[1] - 512) < 1
    t(obs(), priv(tic=100 + 2 * 35 + 4))
    assert t.mode == 'patrol'


def test_respawn_clears_the_fight_keeps_the_map():
    t = DMTeacher(DM)
    t(obs(first=True), priv(labels=[bot(1, 0.0, 400)], objects=[MapObject(1, 'Medikit', 600.0, 700.0, 0.0)]))
    t(obs(first=True), priv(px=100.0, py=100.0))
    assert t.last_seen is None and t.target is None and len(t.known) == 1


def test_unreachable_goal_falls_back_to_patrol():
    t = DMTeacher(DM)
    t(obs(first=True), priv(health=30.0, objects=[MapObject(1, 'Medikit', 2064.0, 2064.0, 0.0)]))
    assert t.mode == 'patrol'


def test_navigation_moves_and_turns_together():
    t = DMTeacher(DM)
    move, turn, _, _ = t(obs(first=True), priv(angle=20.0, objects=[MapObject(1, 'Chaingun', 700.0, 512.0, 0.0)],
                                               weapon=2, owned=(0, 1, 1, 0, 0, 0, 0, 0)))
    assert t.mode == 'gear' and move in (FORWARD, FORWARD_LEFT, FORWARD_RIGHT) and TURN_DEG[turn] < 0


def test_keeps_rocket_range_by_backing_off():
    move, _, _, _ = DMTeacher(DM)(obs(), priv(labels=[bot(1, 0.0, 140)], owned=(0, 1, 1, 0, 0, 1, 0, 0)))
    assert move in (BACK, BACK_LEFT, BACK_RIGHT)


@pytest.mark.vizdoom
def test_teacher_patrols_map02_without_getting_stuck():
    from doomagent.dm.env import DeathmatchEnv
    from doomagent.dm.evaluate import run_match
    e = DeathmatchEnv(replace(DM, bots=0, timelimit=0.5))
    t = DMTeacher(DM)
    r = run_match(e, t, 35000)
    e.close()
    assert r['stuck_share'] < 0.05 and r['path'] > 1500 and r['weapon_share'][2] < 0.5


def test_why_names_the_reason():
    t = DMTeacher(DM)
    t(obs(), priv(labels=[bot(1, 0.0, 600)]))
    assert t.why == 'shoot'
    t(obs(), priv(labels=[bot(1, 40.0, 600)], tic=108))
    assert t.why == 'not_aimed'
    t(obs(), priv(tic=200))
    assert t.why == 'no_combat'


def test_hitscan_fires_within_its_tolerance():
    p = dict(weapon=4, weapon_ammo=50.0, ammo=50.0, owned=(0, 1, 1, 0, 1, 0, 0, 0), slot_ammo=(0, 0, 50, 0, 50, 0, 0, 0))
    assert DMTeacher(DM)(obs(), priv(labels=[bot(1, 2.0, 600)], **p))[2] == 0  # 2° off a ±1.5° target
    assert DMTeacher(replace(DM, hitscan_tol=3.0))(obs(), priv(labels=[bot(1, 2.0, 600)], **p))[2] == 1
    assert DMTeacher(replace(DM, hitscan_tol=3.0))(obs(), priv(labels=[bot(1, 2.0, 600)]))[2] == 0  # not rockets
