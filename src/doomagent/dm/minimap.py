import math

import cv2
import numpy as np

from doomagent.dm.config import BOT, ITEM_KIND
from doomagent.dm.nav import walls_from_sectors

# BGR, dark theme; the agent never sees this map
BG, LINE, WALL, PLAYER = (25, 26, 26), (70, 70, 70), (200, 200, 200), (255, 255, 255)
BOT_COLOR, PATH = (103, 103, 230), (38, 89, 217)
KIND = {'health': (112, 158, 25), 'ammo': (0, 133, 201), 'armor': (233, 133, 144)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


class MapView:
    # top-down map for viewers: east is right, north is up
    def __init__(self, sectors, size=(320, 320), margin=10):
        self.w, self.h = size
        self.m = margin
        pts = np.array([(x, y) for _, _, lines in sectors for x1, y1, x2, y2, _ in lines
                        for x, y in ((x1, y1), (x2, y2))], np.float64)
        self.xmin, self.ymin = pts.min(0)
        self.xmax, self.ymax = pts.max(0)
        self.s = min((self.w - 2 * margin) / (self.xmax - self.xmin),
                     (self.h - 2 * margin) / (self.ymax - self.ymin))
        self.sectors, self._base = sectors, None

    def to_px(self, x, y):
        return (int(round(self.m + (x - self.xmin) * self.s)),
                int(round(self.m + (self.ymax - y) * self.s)))

    def base(self):
        if self._base is None:
            img = np.full((self.h, self.w, 3), BG, np.uint8)
            for _, _, lines in self.sectors:
                for x1, y1, x2, y2, _ in lines:
                    cv2.line(img, self.to_px(x1, y1), self.to_px(x2, y2), LINE, 1, cv2.LINE_AA)
            for x1, y1, x2, y2 in walls_from_sectors(self.sectors):
                cv2.line(img, self.to_px(x1, y1), self.to_px(x2, y2), WALL, 1, cv2.LINE_AA)
            self._base = img
        return self._base

    def draw(self, priv, teacher=None):
        img = self.base().copy()
        for o in priv.objects:
            p = self.to_px(o.x, o.y)
            if o.name == BOT:
                cv2.circle(img, p, 4, BOT_COLOR, -1, cv2.LINE_AA)
            elif o.name in ITEM_KIND:
                cv2.circle(img, p, 3, KIND[ITEM_KIND[o.name]], 1, cv2.LINE_AA)
        if teacher is not None:
            for kind, x, y in teacher.known:
                cv2.circle(img, self.to_px(x, y), 3, KIND[kind], -1, cv2.LINE_AA)
            if teacher.path:
                pts = np.array([self.to_px(x, y) for x, y in teacher.path], np.int32)
                cv2.polylines(img, [pts], False, PATH, 1, cv2.LINE_AA)
            if teacher.goal is not None and teacher.mode != 'combat':
                cv2.circle(img, self.to_px(*teacher.goal), 6, PATH, 1, cv2.LINE_AA)
            cv2.putText(img, teacher.mode, (6, self.h - 8), FONT, 0.45, PLAYER, 1, cv2.LINE_AA)
        p = self.to_px(priv.px, priv.py)
        for d in (-45.0, 45.0):  # the 90 degree view
            a = math.radians(priv.angle + d)
            q = self.to_px(priv.px + 160 * math.cos(a), priv.py + 160 * math.sin(a))
            cv2.line(img, p, q, LINE, 1, cv2.LINE_AA)
        a = math.radians(priv.angle)
        cv2.line(img, p, self.to_px(priv.px + 90 * math.cos(a), priv.py + 90 * math.sin(a)),
                 PLAYER, 1, cv2.LINE_AA)
        cv2.circle(img, p, 4, PLAYER, -1)
        return img
