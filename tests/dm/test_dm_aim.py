from collections import deque

import numpy as np
import pytest

from doomagent.dm.actions import STRAFE_LEFT, STRAFE_RIGHT, TURN_DEG
from doomagent.dm.aim import dodge_move, half_width_deg, lead_point, pick_turn, target_angle


def test_target_angle_left_is_positive():
    assert target_angle(0, 0, 0.0, 0, 100) == pytest.approx(90.0)
    assert target_angle(0, 0, 90.0, 100, 0) == pytest.approx(-90.0)


def test_pick_turn_never_overshoots():
    for rel in np.linspace(-45, 45, 901):
        d = TURN_DEG[pick_turn(rel, 0.5)]
        assert d * rel >= 0 and abs(rel - d) <= abs(rel)   # never the wrong way, never farther from the target


def test_two_steps_reach_any_target():
    # everything on screen (±45°) in two steps to 1° (the turn set was chosen by this before the plan)
    for rel in np.linspace(-45, 45, 1801):
        r = rel - TURN_DEG[pick_turn(rel, 0.5)]
        r -= TURN_DEG[pick_turn(r, 0.5)]
        assert abs(r) <= 1.0


def test_pick_turn_holds_when_aimed():
    assert pick_turn(0.3, 0.5) == 0 and TURN_DEG[pick_turn(44.0, 0.5)] == 40.0


def test_half_width():
    assert half_width_deg(16.0) == pytest.approx(45.0) and half_width_deg(1000.0) < 1.0


def test_lead_point_hitscan_and_rocket():
    track = deque([(0, 500.0, 0.0), (4, 500.0, 40.0)])  # 10 units per tic up
    assert lead_point(track, 0, 0, 0.0) == (500.0, 40.0)
    x, y = lead_point(track, 0, 0, 20.0)
    assert x == 500.0 and y > 200.0


def test_lead_point_still_target():
    assert lead_point(deque([(0, 300.0, 50.0)]), 0, 0, 20.0) == (300.0, 50.0)


def test_dodge_ignores_own_rocket_flying_away():
    assert dodge_move([(100.0, 0.0, 20.0, 0.0)], 0, 0, 0) is None
    assert dodge_move([(400.0, 0.0, -20.0, 0.0)], 0, 0, 0) in (STRAFE_LEFT, STRAFE_RIGHT)


def test_dodge_steps_away_from_the_line():
    # a rocket from the east passing a little to the player's left (north): step right (south)
    assert dodge_move([(400.0, 20.0, -20.0, 0.0)], 0, 0, 0) == STRAFE_RIGHT
    assert dodge_move([(400.0, -20.0, -20.0, 0.0)], 0, 0, 0) == STRAFE_LEFT


def test_dodge_ignores_a_wide_miss():
    assert dodge_move([(400.0, 200.0, -20.0, 0.0)], 0, 0, 0) is None
