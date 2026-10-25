"""Live place maps for the dashboard: while the live game plays, where the player has been (visits per CELL) and the
GRU units most tied to the place (mean activation per cell; a unit's share of its variance explained by the cell),
accumulated over the matches of one network and drawn on the MAP02 outline.
"""
import cv2
import numpy as np

from doomagent.dm.minimap import MapView

CELL = 64.0
MIN_VISITS = 3
LOW, HIGH = np.array([70, 45, 25], np.float32), np.array([255, 190, 80], np.float32)  # BGR: dark navy -> bright blue


class PlaceMaps:
    def __init__(self, sectors, units=512, size=(300, 300)):
        self.view = MapView(sectors, size=size)
        v = self.view
        self.x0, self.y0 = v.xmin, v.ymin
        self.nx = int(np.ceil((v.xmax - v.xmin) / CELL)) + 1
        self.ny = int(np.ceil((v.ymax - v.ymin) / CELL)) + 1
        self.units = units
        self.reset()

    def reset(self):
        self.visits = np.zeros((self.ny, self.nx))
        self.sums = np.zeros((self.ny, self.nx, self.units))
        self.total = np.zeros(self.units)
        self.total_sq = np.zeros(self.units)
        self.steps = 0

    def add(self, x, y, h=None):
        # one step at (x, y); h: the GRU state (units,), or None for a player without memory
        j = min(max(int((x - self.x0) // CELL), 0), self.nx - 1)
        i = min(max(int((y - self.y0) // CELL), 0), self.ny - 1)
        self.visits[i, j] += 1
        self.steps += 1
        if h is not None:
            h = np.asarray(h, np.float64).reshape(-1)
            self.sums[i, j] += h
            self.total += h
            self.total_sq += h * h

    def top_units(self, k=3):
        # (unit, share of its variance explained by the cell) of the k most place-bound units
        ok = self.visits >= MIN_VISITS
        n = self.visits[ok].sum()
        if n < 50 or not self.total_sq.any():
            return []
        mean = self.total / self.steps
        var = self.total_sq / self.steps - mean ** 2
        cell_mean = self.sums[ok] / self.visits[ok][:, None]
        between = (self.visits[ok][:, None] * (cell_mean - mean) ** 2).sum(0) / n
        info = between / np.maximum(var, 1e-8)
        best = np.argsort(-info)[:k]
        return [(int(u), float(info[u])) for u in best]

    def _panel(self, values, title):
        v = self.view
        img = v.base().copy()
        over = img.copy()
        ok = ~np.isnan(values)
        if ok.any():
            lo, hi = np.nanmin(values), np.nanmax(values)
            for i, j in zip(*np.nonzero(ok)):
                t = 0.0 if hi <= lo else (values[i, j] - lo) / (hi - lo)
                color = tuple(int(c) for c in LOW + (HIGH - LOW) * t)
                p0 = v.to_px(self.x0 + j * CELL, self.y0 + (i + 1) * CELL)
                p1 = v.to_px(self.x0 + (j + 1) * CELL, self.y0 + i * CELL)
                cv2.rectangle(over, p0, p1, color, -1)
        img = cv2.addWeighted(over, 0.85, img, 0.15, 0)
        cv2.putText(img, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        return img

    def render(self):
        # visits, then the three most place-bound units, side by side (BGR)
        visits = np.where(self.visits > 0, np.log1p(self.visits), np.nan)
        panels = [self._panel(visits, f'visits: {self.steps} steps')]
        ok = self.visits >= MIN_VISITS
        for u, info in self.top_units(3):
            m = np.full(self.visits.shape, np.nan)
            m[ok] = self.sums[ok][:, u] / self.visits[ok]
            panels.append(self._panel(m, f'unit {u}: {min(info, 1.0):.0%} by place'))
        while len(panels) < 4:
            panels.append(self._panel(np.full(self.visits.shape, np.nan), 'units: after a few hundred steps'))
        return np.concatenate(panels, 1)
