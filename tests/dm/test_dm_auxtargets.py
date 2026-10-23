import numpy as np

from doomagent.dm.auxtargets import (ENEMY_X_BINS, ITEM_KINDS, N_POS, aux_step, heading_bin, mirror_x, pos_cell)
from doomagent.dm.config import BOT, DM
from doomagent.dm.env import DMPrivileged, Label


def _label(oid, name, x, w, wx=500.0, wy=0.0):
    return Label(1, oid, name, x, 100.0, w, 20.0, wx, wy)


def test_enemy_bin_is_the_nearest_visible_bot():
    near = _label(1, BOT, 300.0, 10.0, wx=200.0)       # right edge of a 320-wide screen
    far = _label(2, BOT, 0.0, 10.0, wx=900.0)           # left edge
    hidden = _label(3, BOT, 150.0, 10.0, wx=100.0)      # behind a wall: no pixels
    p = DMPrivileged(labels=[far, near, hidden], vis_px={1: 400, 2: 400, 3: 0})
    x, items, depth = aux_step(p, DM)
    assert x == ENEMY_X_BINS - 1 and not items.any() and depth is None


def test_no_enemy_and_items_in_view():
    p = DMPrivileged(labels=[_label(4, 'Medikit', 100.0, 10.0), _label(5, 'Chaingun', 200.0, 10.0),
                             _label(6, 'Clip', 50.0, 2.0)], vis_px={4: 300, 5: 300, 6: 3})
    x, items, _ = aux_step(p, DM)
    assert x == -1
    assert dict(zip(ITEM_KINDS, items.tolist())) == dict(weapon=True, ammo=False, health=True, armor=False)


def test_coarse_depth_when_the_env_reads_depth():
    p = DMPrivileged(depth=np.full((8, 16), 7, np.uint8))
    assert aux_step(p, DM)[2].shape == (8, 16)


def test_pos_cell_and_heading_bin():
    assert pos_cell(-704.0, -672.0) == 0
    assert 0 <= pos_cell(1951.0, 1599.0) == N_POS - 1
    assert pos_cell(5000.0, 0.0) == -1  # off the map
    assert heading_bin(0.0) == 0 and heading_bin(359.0) == 0 and heading_bin(90.0) == 2 and heading_bin(-90.0) == 6


def test_mirror_x_flips_the_bins():
    assert mirror_x(np.array([0, 7, -1, 3])).tolist() == [7, 0, -1, 4]
