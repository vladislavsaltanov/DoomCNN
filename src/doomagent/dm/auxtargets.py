"""Auxiliary targets: what the engine knows about a step, for extra heads that teach the network to see and to
know where it is. The network never gets them as input; they only shape its features during training.

Per step: the nearest visible bot's place on screen (ENEMY_X_BINS columns, -1 none), item kinds in view,
a coarse depth map (when the env reads depth); from the pose: the map cell and the heading.
(Not 'aux.py': AUX is a reserved file name on Windows.)
"""
import numpy as np

from doomagent.dm.config import ITEM_KIND
from doomagent.dm.teacher import visible_bots

ENEMY_X_BINS = 8
ITEM_KINDS = ('weapon', 'ammo', 'health', 'armor')
HEADING_BINS = 8
POS_ORIGIN, POS_CELL, POS_NX, POS_NY = (-704.0, -672.0), 256.0, 11, 9  # MAP02 bounds in 256-unit cells
N_POS = POS_NX * POS_NY
DEPTH_SHAPE = (8, 16)
AUX_SIZES = dict(enemy=1, enemy_x=ENEMY_X_BINS, items=len(ITEM_KINDS), pos=N_POS, heading=HEADING_BINS,
                 depth=DEPTH_SHAPE[0] * DEPTH_SHAPE[1])


def aux_step(priv, cfg):
    # (enemy column or -1, items in view (4,) bool, coarse depth (8, 16) uint8 or None); bots and items count
    # as seen from cfg.min_vis_px pixels, like the teacher
    bots = visible_bots(priv.labels, priv.px, priv.py, cfg.width, priv.vis_px, cfg.min_vis_px)
    x = -1
    if bots:
        near = min(bots, key=lambda e: e.dist)
        x = int(np.clip((near.offset + 1.0) / 2.0 * ENEMY_X_BINS, 0, ENEMY_X_BINS - 1))
    items = np.zeros(len(ITEM_KINDS), bool)
    for lb in priv.labels:
        kind = ITEM_KIND.get(lb.object_name)
        if kind and (priv.vis_px or {}).get(int(lb.object_id), np.inf) >= cfg.min_vis_px:
            items[ITEM_KINDS.index(kind)] = True
    return x, items, getattr(priv, 'depth', None)


def pos_cell(x, y):
    i, j = int((x - POS_ORIGIN[0]) // POS_CELL), int((y - POS_ORIGIN[1]) // POS_CELL)
    return j * POS_NX + i if 0 <= i < POS_NX and 0 <= j < POS_NY else -1


def heading_bin(angle):
    # 0 = east (angle 0), counter-clockwise, each bin 45 degrees centered on its direction
    return int(((angle + 180.0 / HEADING_BINS) % 360.0) // (360.0 / HEADING_BINS))


def mirror_x(x):
    # enemy columns of a horizontally flipped frame; -1 (no enemy) stays
    x = np.asarray(x)
    return np.where(x >= 0, ENEMY_X_BINS - 1 - x, x)
