import math
from types import SimpleNamespace as NS

import numpy as np

from doomagent.config import SHOOT, TURN_LEFT, TURN_RIGHT
from doomagent.teacher import (Enemy, Teacher, enemies_from_labels, enemy_mask, nearest_aux,
                               teacher_action)


def lb(name, x, w, ox, oy, oid=1):
    return NS(object_name=name, object_id=oid, value=oid, x=x, y=40, width=w, height=20,
              object_position_x=ox, object_position_y=oy)


def E(i, left, right, dist, w=10.0):
    return Enemy(i, left, right, dist, w)


def test_filters_corpses_and_non_monsters():
    labels = [lb('DeadZombieMan', 10, 10, 5, 0), lb('Clip', 50, 5, 3, 0),
              lb('ZombieMan', 70, 20, 4, 0), lb('DoomImpBall', 90, 4, 2, 0)]
    assert len(enemies_from_labels(labels, 0.0, 0.0, 160)) == 1


def test_edges_sign_and_world_distance():
    e = enemies_from_labels([lb('ZombieMan', 0, 20, 3, 4)], 0.0, 0.0, 160)[0]
    assert math.isclose(e.left, -1.0) and math.isclose(e.right, -0.75)
    assert e.offset < 0 and math.isclose(e.dist, 5.0) and e.width_px == 20


def test_no_enemies_keeps_scanning_direction():
    assert teacher_action([], True, scan=TURN_LEFT)[0] == TURN_LEFT
    assert teacher_action([], True, scan=TURN_RIGHT)[0] == TURN_RIGHT


def test_targets_nearest_enemy_by_world_distance():
    far_left, near_right = E(1, -0.9, -0.7, 10.0), E(2, 0.4, 0.6, 2.0)
    assert teacher_action([far_left, near_right], True) == (TURN_RIGHT, 2)


def test_lock_survives_small_distance_changes_only():
    locked, other = E(1, 0.4, 0.6, 2.2), E(2, -0.6, -0.4, 2.0)
    assert teacher_action([locked, other], True, lock_id=1)[0] == TURN_RIGHT
    far_locked = E(1, 0.4, 0.6, 3.0)
    assert teacher_action([far_locked, other], True, lock_id=1)[0] == TURN_LEFT


def test_shoots_only_when_crosshair_in_box_and_ready():
    assert teacher_action([E(1, -0.05, 0.05, 3.0)], True)[0] == SHOOT
    assert teacher_action([E(1, -0.05, 0.05, 3.0)], False)[0] != SHOOT
    assert teacher_action([E(1, 0.02, 0.2, 3.0)], True)[0] == TURN_RIGHT


def test_margin_requires_crosshair_near_the_box_center():
    e = E(1, -0.01, 0.09, 3.0)
    assert teacher_action([e], True, margin=0.2)[0] == TURN_RIGHT
    assert teacher_action([e], True, margin=0.0)[0] == SHOOT


def test_after_a_shot_turns_to_the_next_target():
    enemies = [E(1, -0.05, 0.05, 2.0), E(2, -0.7, -0.5, 5.0)]
    assert teacher_action(enemies, False, skip_id=1) == (TURN_LEFT, 2)


def test_narrow_targets_are_tracked_but_not_shot():
    e = E(1, -0.02, 0.02, 3.0, w=2.0)
    assert teacher_action([e], True, min_w=4.0)[0] != SHOOT
    assert teacher_action([e], True, min_w=0.0)[0] == SHOOT


def test_nearest_aux():
    assert nearest_aux([]) == (0, 0.0)
    assert nearest_aux([E(1, 0.2, 0.4, 9.0), E(2, -0.4, 0.0, 1.0)]) == (1, -0.2)


def test_teacher_scans_towards_the_last_turn():
    t = Teacher()
    left_enemy = NS(labels=[lb('ZombieMan', 10, 10, 5, 0)], attack_ready=True, px=0, py=0,
                    labels_buffer=None)
    empty = NS(labels=[], attack_ready=True, px=0, py=0, labels_buffer=None)
    assert t(None, left_enemy) == TURN_LEFT
    assert t(None, empty) == TURN_LEFT
    t.reset()
    assert t(None, empty) == TURN_RIGHT


def test_enemy_mask_pools_full_resolution_blocks():
    buf = np.zeros((8, 8), np.uint8)
    buf[5, 6] = 3
    buf[0, 0] = 4
    priv = NS(labels_buffer=buf, labels=[lb('ZombieMan', 0, 1, 0, 0, oid=3),
                                         lb('DeadZombieMan', 0, 1, 0, 0, oid=4)])
    m = enemy_mask(priv, NS(height=2, width=2))
    assert m.tolist() == [[False, False], [False, True]]
