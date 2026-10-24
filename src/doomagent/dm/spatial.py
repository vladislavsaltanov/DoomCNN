"""Spatial targets: what lies around the player and how the picture changes when it moves. Like dm.auxtargets
they only shape the network's features in training; the network never gets them as input.

- egomap: walkable cells around the player (MAP_N x MAP_N cells of MAP_CELL units, the player in the middle,
  heading up, its left on the left), from the map geometry the teacher plans on; the half behind is never on
  screen, so it has to be remembered or known from the place
- odometry: how far the player went forward and left and how much it turned over the last ODO_K steps, in its
  own frame then: the motion between the frames of the stack (pressing forward against a wall gives none)
- future: the same motion over the next FUT_K steps and the coarse depth after them, given the moves and turns
  played then: what the picture becomes if the player does this
- goal: where the teacher is going (direction in GOAL_BINS around the heading, distance class) and its mode

The recorder writes egomap and goal (the teacher knows them); the rest comes from the pose, the depth and the
executed actions every file has.
"""
import numpy as np

from doomagent.dm.actions import HEAD_SIZES
from doomagent.dm.auxtargets import DEPTH_SHAPE
from doomagent.dm.config import MODES

MAP_N, MAP_CELL = 16, 32.0
ODO_K = (1, 3)            # the frame stack spans 3 steps of motion
FUT_K = 4
STEP_UNITS, STEP_DEG = 32.0, 30.0  # about one step of walking and of turning: motion over k steps / (k x these)
JUMP = 96.0               # map units per step: more is a teleport, not motion
GOAL_BINS = 16
GOAL_DIST_EDGES = (256.0, 512.0, 1024.0)
FUT_IN = FUT_K * (HEAD_SIZES[0] + HEAD_SIZES[1])  # the moves and turns played, one-hot
FUT_OUT = 3 + DEPTH_SHAPE[0] * DEPTH_SHAPE[1]     # motion after them, coarse depth after them
SPATIAL_SIZES = dict(egomap=MAP_N * MAP_N, odo=3 * len(ODO_K), goal_dir=GOAL_BINS,
                     goal_dist=len(GOAL_DIST_EDGES) + 1, mode=len(MODES))
MOTION_MIRROR = np.array([1.0, -1.0, -1.0], np.float32)  # forward, left, turn of a flipped world

_R = ((MAP_N - 1) / 2.0 - np.arange(MAP_N)) * MAP_CELL  # row 0 is the farthest ahead, column 0 the farthest left
_FWD, _LEFT = np.meshgrid(_R, _R, indexing='ij')


def egomap(nav, x, y, angle):
    # (MAP_N, MAP_N) bool: the NavGrid cells under the egocentric grid that a player can stand in
    a = np.radians(angle)
    c, s = np.cos(a), np.sin(a)
    wx, wy = x + _FWD * c - _LEFT * s, y + _FWD * s + _LEFT * c
    i = np.floor((wy - nav.y0) / nav.cell).astype(int)
    j = np.floor((wx - nav.x0) / nav.cell).astype(int)
    ok = (i >= 0) & (i < nav.ny) & (j >= 0) & (j < nav.nx)
    out = np.zeros((MAP_N, MAP_N), bool)
    out[ok] = nav.walk[i[ok], j[ok]]
    return out


def ego_delta(p0, p1):
    # (N, 3) motion from poses p0 to p1 (x, y, angle) in p0's frame: forward, left, turn in degrees (+ is left)
    a = np.radians(p0[:, 2])
    dx, dy = p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1]
    turn = (p1[:, 2] - p0[:, 2] + 180.0) % 360.0 - 180.0
    return np.stack([dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a), turn], 1)


def _motion(pose, life, t0, t1, k):
    # scaled motion from steps t0 to t1, k steps apart, and whether it is motion inside one life
    d = ego_delta(pose[t0], pose[t1])
    ok = (life[t0] == life[t1]) & (np.hypot(d[:, 0], d[:, 1]) <= JUMP * k)
    return (d / np.array([STEP_UNITS * k, STEP_UNITS * k, STEP_DEG * k])).astype(np.float32), ok


def goal_targets(pose, goal):
    # direction bin (0 = straight ahead, counter-clockwise) and distance class of the goal; -1 without one
    none = np.isnan(goal).any(1)
    g = np.where(none[:, None], pose[:, :2] + 1.0, goal)
    dx, dy = g[:, 0] - pose[:, 0], g[:, 1] - pose[:, 1]
    w = 360.0 / GOAL_BINS
    rel = (np.degrees(np.arctan2(dy, dx)) - pose[:, 2] + w / 2.0) % 360.0
    d = (rel // w).astype(np.int64)
    r = np.searchsorted(GOAL_DIST_EDGES, np.hypot(dx, dy)).astype(np.int64)
    d[none], r[none] = -1, -1
    return d, r


def mirror_goal_dir(d):
    # goal direction bins of a flipped world; -1 stays
    return np.where(d >= 0, (-d) % GOAL_BINS, d)


def spatial_targets(z):
    # per-step targets of a recorded file (z: its arrays); a file without egomap or goal has them off
    pose, first = np.asarray(z['pose'], np.float64), np.asarray(z['first'], bool)
    T = len(pose)
    t, life = np.arange(T), np.cumsum(first)
    odo, odo_ok = [], []
    for k in ODO_K:
        d, ok = _motion(pose, life, np.maximum(t - k, 0), t, k)
        odo.append(d)
        odo_ok.append(ok & (t >= k))
    end = np.minimum(t + FUT_K, T - 1)
    fut, fut_ok = _motion(pose, life, t, end, FUT_K)
    played = np.asarray(z['executed'])[np.minimum(t[:, None] + np.arange(FUT_K), T - 1)]
    out = dict(sp_odo=np.concatenate(odo, 1), sp_odo_ok=np.stack(odo_ok, 1), sp_fut=fut,
               sp_fut_ok=fut_ok & (t + FUT_K < T), sp_fut_idx=end, sp_fut_move=played[..., 0].astype(np.int64),
               sp_fut_turn=played[..., 1].astype(np.int64), sp_mode=np.asarray(z['modes'], np.int64))
    names = z.files if hasattr(z, 'files') else z
    has_map = 'egomap' in names
    out['sp_map'] = (np.asarray(z['egomap']).reshape(T, -1) if has_map else np.zeros((T, MAP_N * MAP_N), bool))
    out['sp_map_ok'] = np.full(T, has_map)
    goal = np.asarray(z['goal'], np.float64) if 'goal' in names else np.full((T, 2), np.nan)
    out['sp_goal_dir'], out['sp_goal_dist'] = goal_targets(pose, goal)
    return out
