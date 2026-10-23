import heapq
import math

import numpy as np

from doomagent.dm.config import PLAYER_HEIGHT, PLAYER_RADIUS, STEP_HEIGHT

MISSILE_Z = 32.0  # a player's rocket starts this far above the player's floor and flies level
MISSILE_H = 8.0   # the rocket's height


def walls_from_sectors(sectors, step=STEP_HEIGHT, height=PLAYER_HEIGHT):
    # (K, 4) segments the player cannot cross: blocking lines, one-sided lines, and two-sided lines
    # with a floor step above `step` or an opening lower than `height`
    sides = {}
    for k, (floor, ceil, lines) in enumerate(sectors):
        for x1, y1, x2, y2, blocking in lines:
            key = tuple(sorted([(x1, y1), (x2, y2)]))
            sides.setdefault(key, []).append((k, blocking))
    out = []
    for (a, b), owners in sides.items():
        ks = sorted({k for k, _ in owners})
        wall = any(bl for _, bl in owners) or len(ks) == 1
        if not wall:
            floors = [sectors[k][0] for k in ks]
            ceils = [sectors[k][1] for k in ks]
            wall = max(floors) - min(floors) > step or min(ceils) - max(floors) < height
        if wall:
            out.append((a[0], a[1], b[0], b[1]))
    return np.array(out, np.float64).reshape(-1, 4)


def _point_segment_dist(px, py, w):
    # distances (N, K) from points to segments
    x1, y1, x2, y2 = (w[:, i][None, :] for i in range(4))
    dx, dy = x2 - x1, y2 - y1
    t = np.clip(((px[:, None] - x1) * dx + (py[:, None] - y1) * dy) / np.maximum(dx * dx + dy * dy, 1e-9),
                0.0, 1.0)
    return np.hypot(px[:, None] - (x1 + t * dx), py[:, None] - (y1 + t * dy))


def segments_cross(ax, ay, bx, by, walls):
    # does segment a-b properly intersect any wall
    if len(walls) == 0:
        return False
    x1, y1, x2, y2 = (walls[:, i] for i in range(4))

    def orient(px, py, qx, qy, rx, ry):
        return np.sign((qx - px) * (ry - py) - (qy - py) * (rx - px))

    d1 = orient(x1, y1, x2, y2, ax, ay)
    d2 = orient(x1, y1, x2, y2, bx, by)
    d3 = orient(ax, ay, bx, by, x1, y1)
    d4 = orient(ax, ay, bx, by, x2, y2)
    return bool(np.any((d1 * d2 < 0) & (d3 * d4 < 0)))


def shot_lines_from_sectors(sectors):
    # every line with the highest floor and lowest ceiling of its sides; a one-sided line gets
    # (+inf, -inf), so it stops any rocket
    sides = {}
    for floor, ceil, lines in sectors:
        for x1, y1, x2, y2, _ in lines:
            sides.setdefault(tuple(sorted([(x1, y1), (x2, y2)])), []).append((floor, ceil))
    segs, lo, hi = [], [], []
    for (a, b), fc in sides.items():
        segs.append((a[0], a[1], b[0], b[1]))
        two = len(fc) > 1
        lo.append(max(f for f, _ in fc) if two else np.inf)
        hi.append(min(c for _, c in fc) if two else -np.inf)
    return np.array(segs, np.float64).reshape(-1, 4), np.array(lo), np.array(hi)


def _ray(walls, x, y, angle_deg, max_d):
    if len(walls) == 0:
        return max_d
    dx, dy = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    x1, y1, x2, y2 = (walls[:, i] for i in range(4))
    ex, ey = x2 - x1, y2 - y1
    den = dx * ey - dy * ex
    with np.errstate(divide='ignore', invalid='ignore'):
        t = ((x1 - x) * ey - (y1 - y) * ex) / den
        u = ((x1 - x) * dy - (y1 - y) * dx) / den
    ok = (np.abs(den) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
    return float(min(t[ok].min(), max_d)) if ok.any() else max_d


class NavGrid:
    # a walkable grid over the map; built once per map (nav_for caches it)
    def __init__(self, sectors, cell=16.0, clearance=PLAYER_RADIUS + 2):
        self.cell = cell
        self.walls = walls_from_sectors(sectors)
        self.shot_lines, self.shot_lo, self.shot_hi = shot_lines_from_sectors(sectors)
        pts = np.array([(x, y) for _, _, lines in sectors for x1, y1, x2, y2, _ in lines
                        for x, y in ((x1, y1), (x2, y2))], np.float64)
        self.x0, self.y0 = pts[:, 0].min(), pts[:, 1].min()
        self.nx = int(math.ceil((pts[:, 0].max() - self.x0) / cell))
        self.ny = int(math.ceil((pts[:, 1].max() - self.y0) / cell))
        jj, ii = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        cx, cy = self.x0 + (jj.ravel() + 0.5) * cell, self.y0 + (ii.ravel() + 0.5) * cell
        floor = np.full(cx.shape, np.nan)
        gap = np.zeros(cx.shape)
        for f, c, lines in sectors:
            inside = np.zeros(cx.shape, bool)
            for x1, y1, x2, y2, _ in lines:
                if y1 == y2:
                    continue
                cross = ((y1 > cy) != (y2 > cy)) & (cx < (x2 - x1) * (cy - y1) / (y2 - y1) + x1)
                inside ^= cross
            floor[inside], gap[inside] = f, c - f
        walk = ~np.isnan(floor) & (gap >= PLAYER_HEIGHT)
        if len(self.walls) and walk.any():
            idx = np.flatnonzero(walk)
            near = _point_segment_dist(cx[idx], cy[idx], self.walls).min(axis=1) < clearance
            walk[idx[near]] = False
        self.floor = floor.reshape(self.ny, self.nx).astype(np.float32)
        self.walk = walk.reshape(self.ny, self.nx)

    def cell_of(self, x, y):
        i = int(np.clip((y - self.y0) // self.cell, 0, self.ny - 1))
        j = int(np.clip((x - self.x0) // self.cell, 0, self.nx - 1))
        return i, j

    def center(self, i, j):
        return self.x0 + (j + 0.5) * self.cell, self.y0 + (i + 0.5) * self.cell

    def nearest_walkable(self, i, j):
        if self.walk[i, j]:
            return i, j
        ii, jj = np.nonzero(self.walk)
        k = int(np.argmin((ii - i) ** 2 + (jj - j) ** 2))
        return int(ii[k]), int(jj[k])

    def path(self, start_xy, goal_xy):
        if not self.walk.any():
            return []
        s = self.nearest_walkable(*self.cell_of(*start_xy))
        g = self.nearest_walkable(*self.cell_of(*goal_xy))
        walk, floor = self.walk, self.floor
        h = lambda a: max(abs(a[0] - g[0]), abs(a[1] - g[1])) + 0.41421356 * min(abs(a[0] - g[0]), abs(a[1] - g[1]))  # noqa: E731
        best, came, todo = {s: 0.0}, {s: None}, [(h(s), 0.0, s)]
        while todo:
            _, cost, a = heapq.heappop(todo)
            if a == g:
                out = []
                while a is not None:
                    out.append(self.center(*a))
                    a = came[a]
                return out[::-1]
            if cost > best[a]:
                continue
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if not di and not dj:
                        continue
                    b = (a[0] + di, a[1] + dj)
                    if not (0 <= b[0] < self.ny and 0 <= b[1] < self.nx) or not walk[b]:
                        continue
                    if abs(floor[b] - floor[a]) > STEP_HEIGHT:
                        continue
                    if di and dj and not (walk[a[0] + di, a[1]] and walk[a[0], a[1] + dj]):
                        continue
                    c = cost + (1.41421356 if di and dj else 1.0)
                    if c < best.get(b, float('inf')):
                        best[b], came[b] = c, a
                        heapq.heappush(todo, (c + h(b), c, b))
        return []

    def wall_dist(self, x, y, angle_deg, max_d=4096.0):
        # to the nearest line the player cannot walk through
        return _ray(self.walls, x, y, angle_deg, max_d)

    def shot_dist(self, x, y, angle_deg, max_d=4096.0):
        # to the nearest line that stops a rocket fired from (x, y): one-sided lines, and two-sided ones
        # where a floor rises above the rocket or a ceiling comes down on it; the impassable flag and
        # ledges going down do not stop it
        f = self.floor[self.cell_of(x, y)]
        rz = (0.0 if np.isnan(f) else float(f)) + MISSILE_Z
        stop = (self.shot_lo > rz) | (self.shot_hi < rz + MISSILE_H)
        return _ray(self.shot_lines[stop], x, y, angle_deg, max_d)

    def patrol_ring(self, k=8, seed=0):
        ii, jj = np.nonzero(self.walk)
        pts = np.stack([self.x0 + (jj + 0.5) * self.cell, self.y0 + (ii + 0.5) * self.cell], 1)
        rng = np.random.default_rng(seed)
        centers = [pts[rng.integers(len(pts))]]
        for _ in range(1, k):  # k-means++ seeding
            d = np.min([((pts - c) ** 2).sum(1) for c in centers], axis=0)
            centers.append(pts[rng.choice(len(pts), p=d / d.sum())])
        centers = np.array(centers)
        for _ in range(25):
            lab = np.argmin(((pts[:, None, :] - centers[None]) ** 2).sum(2), axis=1)
            for c in range(k):
                if (lab == c).any():
                    centers[c] = pts[lab == c].mean(0)
        snapped = [tuple(float(v) for v in pts[np.argmin(((pts - c) ** 2).sum(1))]) for c in centers]
        mx, my = np.mean(snapped, axis=0)
        return sorted(snapped, key=lambda p: math.atan2(p[1] - my, p[0] - mx))


_CACHE = {}


def nav_for(sectors):
    key = hash(tuple(sectors))
    if key not in _CACHE:
        _CACHE[key] = NavGrid(sectors)
    return _CACHE[key]


def lookahead(path, x, y, dist=64.0):
    if not path:
        return None
    k = min(range(len(path)), key=lambda i: (path[i][0] - x) ** 2 + (path[i][1] - y) ** 2)
    for p in path[k:]:
        if math.hypot(p[0] - x, p[1] - y) >= dist:
            return p
    return path[-1]
