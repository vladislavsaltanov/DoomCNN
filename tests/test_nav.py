import math

import pytest

from doomagent.dm.nav import NavGrid, lookahead, segments_cross


def sector(points, floor=0.0, ceil=128.0, open_edges=()):
    n = len(points)
    return (floor, ceil, tuple((*points[i], *points[(i + 1) % n], i not in open_edges) for i in range(n)))


BOX = [sector([(0, 0), (512, 0), (512, 512), (0, 512)])]
LEFT = sector([(0, 0), (192, 0), (192, 384), (192, 448), (192, 512), (0, 512)], open_edges=(2,))
CORR = sector([(192, 384), (320, 384), (320, 448), (192, 448)], open_edges=(1, 3))
RIGHT = sector([(320, 0), (512, 0), (512, 512), (320, 512), (320, 448), (320, 384)], open_edges=(4,))


def test_open_room_path_is_straight():
    p = NavGrid(BOX).path((64, 64), (448, 448))
    assert p and max(abs(x - y) for x, y in p) <= 16


def test_path_goes_through_the_corridor():
    g = NavGrid([LEFT, CORR, RIGHT])
    p = g.path((96, 96), (416, 96))
    assert p and all(384 < y < 448 for x, y in p if 192 < x < 320)
    assert not any(segments_cross(*a, *b, g.walls) for a, b in zip(p, p[1:]))


@pytest.mark.parametrize('floor, reachable', [(40.0, False), (16.0, True)])
def test_step_height(floor, reachable):
    a = sector([(0, 0), (256, 0), (256, 256), (0, 256)], open_edges=(1,))
    b = sector([(256, 0), (512, 0), (512, 256), (256, 256)], floor=floor, open_edges=(3,))
    assert bool(NavGrid([a, b]).path((64, 128), (448, 128))) == reachable


def test_low_ceiling_is_not_walkable():
    g = NavGrid([sector([(0, 0), (256, 0), (256, 256), (0, 256)], floor=264.0, ceil=268.0)])
    assert not g.walk.any()


def test_goal_inside_wall_snaps_to_nearest_walkable():
    g = NavGrid(BOX)
    p = g.path((256, 256), (2, 256))
    assert p and math.hypot(p[-1][0] - 2, p[-1][1] - 256) <= 32 and g.walk[g.cell_of(*p[-1])]


def test_wall_dist_in_a_box():
    g = NavGrid(BOX)
    for angle in (0.0, 90.0, 180.0, 270.0):
        assert abs(g.wall_dist(256, 256, angle) - 256) < 1


def test_lookahead():
    path = [(16.0 * k, 0.0) for k in range(11)]
    assert lookahead(path, 0, 0, 64) == (64.0, 0.0)
    assert lookahead(path, 500, 0, 64) == path[-1]


def test_patrol_ring_is_walkable_spread_and_deterministic():
    g = NavGrid([LEFT, CORR, RIGHT])
    ring = g.patrol_ring(8)
    assert ring == g.patrol_ring(8) and len(set(ring)) == 8
    assert all(g.walk[g.cell_of(x, y)] for x, y in ring)


@pytest.mark.vizdoom
def test_cig_grid_and_paths():
    from dataclasses import replace
    from doomagent.dm.config import DM
    from doomagent.dm.env import DeathmatchEnv
    e = DeathmatchEnv(replace(DM, bots=0, timelimit=0.1))
    e.reset(35000)
    g = NavGrid(e.sectors)
    e.close()
    assert 0.2 < g.walk.mean() < 0.9
    ring = g.patrol_ring(8)
    p = g.path(ring[0], ring[4])
    assert p and not any(segments_cross(*a, *b, g.walls) for a, b in zip(p, p[1:]))
