"""The teacher's navigation grid on cig MAP01: walkable cells, walls, the patrol ring and one A* path.

    python scripts/dm/dm_navgrid_png.py --out artifacts/dm/d1_navgrid.png
"""
import argparse
from dataclasses import replace

import cv2
import numpy as np

from doomagent.dm.config import DM, VAL_SEEDS
from doomagent.dm.env import DeathmatchEnv
from doomagent.dm.minimap import PATH, MapView
from doomagent.dm.nav import NavGrid

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='artifacts/dm/d1_navgrid.png')
a = ap.parse_args()
env = DeathmatchEnv(replace(DM, bots=0, timelimit=0.1))
env.reset(VAL_SEEDS[0])
sectors = env.sectors
env.close()
g = NavGrid(sectors)
m = MapView(sectors, size=(760, 760), margin=20)
img = m.base().copy()
cell = int(round(g.cell * m.s / 2))
for i, j in zip(*np.nonzero(g.walk)):
    x, y = m.to_px(*g.center(i, j))
    cv2.rectangle(img, (x - cell, y - cell), (x + cell, y + cell), (52, 54, 54), -1)
img = np.maximum(img, m.base())  # walls on top of the cells
ring = g.patrol_ring()
path = g.path(ring[0], ring[4])
cv2.polylines(img, [np.array([m.to_px(x, y) for x, y in path], np.int32)], False, PATH, 2, cv2.LINE_AA)
for k, (x, y) in enumerate(ring):
    p = m.to_px(x, y)
    cv2.circle(img, p, 9, (229, 135, 57), -1, cv2.LINE_AA)
    cv2.putText(img, str(k), (p[0] - 5, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA)
cv2.putText(img, f'walkable cells {g.walk.sum()} of {g.walk.size} (16 units); patrol ring 0-7; '
            'A* path 0 -> 4', (20, 752), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
cv2.imwrite(a.out, img)
print('written', a.out, 'path cells', len(path))
