"""Deathmatch v2 actions: four heads chosen together every step (move, turn, fire, next weapon), the
mirror of an action for flipped frames, the agent's vector and the buttons a step presses.

An action is a tuple (move, turn, fire, weapon) of value indices; files keep them as (T, 4) int64.
"""
import numpy as np

from doomagent.dm.config import AMMO_MAX
from doomagent.env import DELTA_SIGN

STAY, FORWARD, BACK, STRAFE_LEFT, STRAFE_RIGHT, FORWARD_LEFT, FORWARD_RIGHT, BACK_LEFT, BACK_RIGHT = range(9)
MOVE_NAMES = ('stay', 'forward', 'back', 'strafe_left', 'strafe_right', 'forward_left', 'forward_right',
              'back_left', 'back_right')
# degrees per step, + is left; any target on screen (±45°) in two steps to 1°, in one step to 5°
TURN_DEG = (0.0, 1.0, -1.0, 3.0, -3.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 40.0, -40.0)
HEADS = ('move', 'turn', 'fire', 'weapon')
HEAD_SIZES = (len(MOVE_NAMES), len(TURN_DEG), 2, 2)
N_LOGITS = sum(HEAD_SIZES)
MOVE_MIRROR = (STAY, FORWARD, BACK, STRAFE_RIGHT, STRAFE_LEFT, FORWARD_RIGHT, FORWARD_LEFT, BACK_RIGHT, BACK_LEFT)
TURN_MIRROR = tuple(TURN_DEG.index(-d) for d in TURN_DEG)
NOOP = (STAY, 0, 0, 0)
N_WEAPON_SLOTS = 7
VEC_DIM = 3 + N_WEAPON_SLOTS

# move value -> (forward, back, left, right)
_MOVE_KEYS = ((0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 1, 0), (1, 0, 0, 1),
              (0, 1, 1, 0), (0, 1, 0, 1))


def mirror_action(a):
    move, turn, fire, weapon = a
    return MOVE_MIRROR[move], TURN_MIRROR[turn], fire, weapon


def turn_index(deg):
    return TURN_DEG.index(float(deg))


def make_vec(health, armor, weapon, weapon_ammo):
    # [health/100, armor/200, ammo of the weapon in hand / its maximum, weapon slot one-hot]; the fist
    # (slot 1) has no ammo
    v = np.zeros(VEC_DIM, np.float32)
    v[0], v[1] = health / 100.0, armor / 200.0
    if weapon > 1:
        v[2] = weapon_ammo / AMMO_MAX[int(weapon)]
    if 1 <= weapon <= N_WEAPON_SLOTS:
        v[2 + int(weapon)] = 1.0
    return v


def buttons_for(action, frame_skip):
    # (buttons of the step's first tic, buttons of the other tics) in the order MOVE_FORWARD, MOVE_BACKWARD,
    # MOVE_LEFT, MOVE_RIGHT, ATTACK, SELECT_NEXT_WEAPON, TURN_LEFT_RIGHT_DELTA; the turn is spread evenly over
    # the tics, the next-weapon key is pressed on the first tic only
    move, turn, fire, weapon = action
    keys = list(_MOVE_KEYS[move]) + [int(fire)]
    delta = -DELTA_SIGN * TURN_DEG[turn] / frame_skip  # a positive delta turns right (measured in E0)
    return keys + [int(weapon), delta], keys + [0, delta]
