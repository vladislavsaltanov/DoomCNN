from types import SimpleNamespace

import numpy as np
import pytest

from doomagent.dm.spatial import (FUT_K, GOAL_BINS, MAP_N, STEP_UNITS, ego_delta, egomap, goal_targets,
                                  mirror_goal_dir, spatial_targets)


def _nav(walk_fn, size=2048.0, cell=16.0):
    # a NavGrid stand-in: walk[i, j] for the cell centers of a square map from (-size/2, -size/2)
    n = int(size / cell)
    x0 = y0 = -size / 2
    cy, cx = np.meshgrid(y0 + (np.arange(n) + 0.5) * cell, x0 + (np.arange(n) + 0.5) * cell, indexing='ij')
    return SimpleNamespace(x0=x0, y0=y0, cell=cell, nx=n, ny=n, walk=walk_fn(cx, cy))


def test_egomap_is_heading_up_with_left_on_the_left():
    nav = _nav(lambda x, y: x < 100.0)  # a wall east of x = 100
    m = egomap(nav, 0.0, 0.0, 0.0)      # facing east: the wall is ahead, at the top rows
    assert m.shape == (MAP_N, MAP_N)
    assert not m[0].any() and m[-1].all()
    m = egomap(nav, 0.0, 0.0, 90.0)     # facing north: east is on the right, the last columns
    assert not m[:, -1].any() and m[:, 0].all()
    assert not egomap(nav, 5000.0, 0.0, 0.0).any()  # off the map nothing is walkable


def test_ego_delta_in_the_frame_of_the_first_pose():
    p0 = np.array([[0.0, 0.0, 90.0]] * 3)
    p1 = np.array([[0.0, 10.0, 90.0], [-10.0, 0.0, 90.0], [0.0, 0.0, 70.0]])
    d = ego_delta(p0, p1)
    assert d[0] == pytest.approx([10.0, 0.0, 0.0], abs=1e-9)    # north is forward
    assert d[1] == pytest.approx([0.0, 10.0, 0.0], abs=1e-9)    # west is left
    assert d[2] == pytest.approx([0.0, 0.0, -20.0], abs=1e-9)   # a right turn is negative
    assert ego_delta(np.array([[0, 0, 350.0]]), np.array([[0, 0, 10.0]]))[0, 2] == pytest.approx(20.0)


def _file(T=30, first_at=(0, 20), step=8.0):
    first = np.zeros(T, bool)
    first[list(first_at)] = True
    pose = np.stack([np.arange(T) * step, np.zeros(T), np.zeros(T)], 1).astype(np.float32)
    ex = np.stack([np.arange(T) % 9, np.arange(T) % 15, np.zeros(T), np.zeros(T)], 1).astype(np.int64)
    return dict(pose=pose, first=first, executed=ex, modes=(np.arange(T) % 5).astype(np.int8))


def test_odometry_over_the_last_steps_inside_a_life():
    s = spatial_targets(_file())
    assert s['sp_odo'][5] == pytest.approx([8 / STEP_UNITS, 0, 0, 8 / STEP_UNITS, 0, 0])
    assert s['sp_odo_ok'][5].all()
    assert not s['sp_odo_ok'][0].any() and not s['sp_odo_ok'][20].any()   # life starts: no motion before
    assert s['sp_odo_ok'][21].tolist() == [True, False]                    # one step into the new life
    assert s['sp_mode'].tolist() == (np.arange(30) % 5).tolist()


def test_future_motion_and_the_actions_played():
    z = _file()
    s = spatial_targets(z)
    assert s['sp_fut_ok'][3] and s['sp_fut'][3] == pytest.approx([8 / STEP_UNITS, 0, 0])
    assert s['sp_fut_idx'][3] == 3 + FUT_K
    assert s['sp_fut_move'][3].tolist() == (np.arange(3, 3 + FUT_K) % 9).tolist()
    assert s['sp_fut_turn'][3].tolist() == (np.arange(3, 3 + FUT_K) % 15).tolist()
    assert not s["sp_fut_ok"][16] and s["sp_fut_ok"][15]   # 16 + 4 is the first step of the next life
    assert not s['sp_fut_ok'][27]                           # past the end of the match


def test_a_teleport_is_not_motion():
    z = _file(first_at=(0,))
    z['pose'][10:, 0] += 3000.0
    s = spatial_targets(z)
    assert not s['sp_odo_ok'][10].any() and s['sp_odo_ok'][9].all() and not s['sp_fut_ok'][8]


def test_spatial_targets_off_in_files_without_them():
    s = spatial_targets(_file())
    assert not s['sp_map_ok'].any() and (s['sp_goal_dir'] == -1).all() and (s['sp_goal_dist'] == -1).all()
    z = _file()
    z['egomap'] = np.ones((30, MAP_N, MAP_N), bool)
    z['goal'] = np.tile([100.0, 0.0], (30, 1)).astype(np.float32)
    s = spatial_targets(z)
    assert s['sp_map_ok'].all() and s['sp_map'].shape == (30, MAP_N * MAP_N) and (s['sp_goal_dir'] >= 0).all()


def test_goal_direction_and_distance():
    pose = np.zeros((5, 3))
    goal = np.array([[300.0, 0.0], [0.0, 600.0], [-2000.0, 0.0], [0.0, -100.0], [np.nan, np.nan]])
    d, r = goal_targets(pose, goal)
    q = GOAL_BINS // 4
    assert d.tolist() == [0, q, 2 * q, 3 * q, -1]   # ahead, left, behind, right, none
    assert r.tolist() == [1, 2, 3, 0, -1]           # classes split at 256, 512, 1024
    assert mirror_goal_dir(np.array([0, q, 2 * q, 3 * q, -1])).tolist() == [0, 3 * q, 2 * q, q, -1]
