"""The cheater teacher's reflexes: the angle to a point, where to aim at a moving target for a given
projectile speed, which turn of the turn head to take, and which way to step out of a projectile's path."""
import math

from doomagent.dm.actions import (BACK, BACK_LEFT, BACK_RIGHT, FORWARD, FORWARD_LEFT, FORWARD_RIGHT, STRAFE_LEFT,
                                  STRAFE_RIGHT, TURN_DEG)
from doomagent.dm.config import PLAYER_RADIUS

_D = math.sqrt(0.5)
# each move in the player's frame: (forward, left)
MOVE_DIRS = {FORWARD: (1.0, 0.0), BACK: (-1.0, 0.0), STRAFE_LEFT: (0.0, 1.0), STRAFE_RIGHT: (0.0, -1.0),
             FORWARD_LEFT: (_D, _D), FORWARD_RIGHT: (_D, -_D), BACK_LEFT: (-_D, _D), BACK_RIGHT: (-_D, -_D)}


def wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def target_angle(px, py, angle, tx, ty):
    # degrees from the view direction to (tx, ty), + is left
    return wrap(math.degrees(math.atan2(ty - py, tx - px)) - angle)


def lead_point(track, px, py, speed):
    # where a shot fired now meets the target: its current spot for hitscan (speed 0) or a still target,
    # else ahead along its velocity from the track's ends ((tic, x, y) entries), refined once by the range
    _, tx, ty = track[-1]
    if speed <= 0.0 or len(track) < 2 or track[-1][0] == track[0][0]:
        return tx, ty
    dt = track[-1][0] - track[0][0]
    vx, vy = (track[-1][1] - track[0][1]) / dt, (track[-1][2] - track[0][2]) / dt
    t = math.hypot(tx - px, ty - py) / speed
    lx, ly = tx + vx * t, ty + vy * t
    t = math.hypot(lx - px, ly - py) / speed
    return tx + vx * t, ty + vy * t


def pick_turn(rel_deg, tol_deg):
    # the turn closest to the angle needed (0 when within tol): with 0 in the set it never turns the wrong
    # way or away from the target; a target on screen (±45°) takes one step to 5° and two to 1°
    if abs(rel_deg) <= tol_deg:
        return 0
    return min(range(len(TURN_DEG)), key=lambda i: (abs(rel_deg - TURN_DEG[i]), abs(TURN_DEG[i])))


def half_width_deg(dist, radius=PLAYER_RADIUS):
    return math.degrees(math.atan(radius / max(dist, 1e-6)))


def dodge_move(projectiles, px, py, angle, threat=64.0):
    # projectiles: (x, y, vx, vy) per tic. The soonest one that comes closer than `threat` on its line
    # gives the move best aligned with the way out of that line; None if nothing threatens
    best = None
    for x, y, vx, vy in projectiles:
        v2 = vx * vx + vy * vy
        rx, ry = px - x, py - y
        along = rx * vx + ry * vy
        if v2 < 1e-6 or along <= 0.0:  # still, or flying away
            continue
        t = along / v2
        cx, cy = x + vx * t, y + vy * t
        miss = math.hypot(px - cx, py - cy)
        if miss < threat and (best is None or t < best[0]):
            ex, ey = (px - cx, py - cy) if miss > 1e-3 else (vy, -vx)  # dead on: either side; pick one
            best = (t, ex, ey)
    if best is None:
        return None
    _, ex, ey = best
    a = math.radians(angle)
    f, lft = ex * math.cos(a) + ey * math.sin(a), -ex * math.sin(a) + ey * math.cos(a)
    return max(MOVE_DIRS, key=lambda m: MOVE_DIRS[m][0] * f + MOVE_DIRS[m][1] * lft)
