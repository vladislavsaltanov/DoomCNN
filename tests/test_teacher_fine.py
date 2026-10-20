from types import SimpleNamespace as NS

from doomagent.config import FINE_LEFT, FINE_RIGHT, TURN_RIGHT
from doomagent.teacher import Enemy, Teacher, teacher_action


def lb(name, x, w, ox, oy, oid=1):
    return NS(object_name=name, object_id=oid, value=oid, x=x, y=40, width=w, height=20,
              object_position_x=ox, object_position_y=oy)


def E(i, left, right, dist, w=10.0):
    return Enemy(i, left, right, dist, w)


def test_fine_turn_when_the_target_is_inside_the_fine_zone():
    assert teacher_action([E(1, 0.02, 0.2, 3.0)], True, fine_zone=0.15)[0] == FINE_RIGHT


def test_coarse_turn_when_the_target_is_outside_the_fine_zone():
    assert teacher_action([E(1, 0.6, 0.8, 3.0)], True, fine_zone=0.15)[0] == TURN_RIGHT


def test_fine_turn_holds_an_aimed_target_while_reloading():
    # aimed but not ready: a fine step keeps the crosshair near the target
    assert teacher_action([E(1, -0.06, 0.02, 3.0)], False, fine_zone=0.15)[0] == FINE_LEFT


def test_teacher_uses_fine_turns_by_default():
    t = Teacher()
    # box at 82..88 px of 160: offset +0.06, inside the default fine zone
    priv = NS(labels=[lb('ZombieMan', 82, 6, 4, 0)], attack_ready=True, px=0, py=0,
              labels_buffer=None)
    assert t(None, priv) == FINE_RIGHT
