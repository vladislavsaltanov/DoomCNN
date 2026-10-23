"""The M1 memory over time, for the viewer: 512 GRU units (rows) over the last 20 s (columns), events
on top, and the mean update gate below (near 1 the memory holds, near 0 it is overwritten)."""
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw

from doomagent.brain import _font
from doomagent.dm.config import BOT

BG, INK, MUTED = (25, 26, 26), (236, 239, 240), (163, 149, 139)
BLUE, GRAY, RED = np.array([229, 135, 57]), np.array([56, 56, 53]), np.array([103, 103, 230])  # BGR
EVENT_TEXT = {'bot_seen': 'бот', 'bot_lost': 'пропал', 'death': 'смерть', 'pickup': 'подбор'}
F_SMALL = _font(12)


def events_from(prev, priv, first, match_start):
    # what happened on this step, from game data (labels only for the viewer)
    if prev is None or match_start:
        return set()
    had = any(lb.object_name == BOT for lb in prev.labels)
    has = any(lb.object_name == BOT for lb in priv.labels)
    ev = set()
    if has and not had:
        ev.add('bot_seen')
    if had and not has:
        ev.add('bot_lost')
    if first:
        ev.add('death')
    elif priv.health > prev.health or priv.ammo > prev.ammo or priv.armor > prev.armor:
        ev.add('pickup')
    return ev


def diverging(v):
    # [-1, 1] -> BGR: blue below 0, gray at 0, red above
    v = np.clip(v, -1, 1)[..., None]
    return np.where(v < 0, GRAY + (BLUE - GRAY) * -v, GRAY + (RED - GRAY) * v).astype(np.uint8)


class MemoryTape:
    def __init__(self, steps=175):
        self.cols, self.zs, self.evs = deque(maxlen=steps), deque(maxlen=steps), deque(maxlen=steps)
        self.steps = steps

    def push(self, h, z, events=()):
        # h and z of the step whose action was taken (DMAgent.explain)
        self.cols.append(np.asarray(h, np.float32))
        self.zs.append(float(np.mean(z)))
        self.evs.append(set(events))

    @property
    def H(self):
        return np.stack(self.cols, 1) if self.cols else np.zeros((512, 0), np.float32)

    @property
    def Z(self):
        return np.array(self.zs)

    @property
    def E(self):
        return list(self.evs)

    def render(self, width=900, height=420):
        img = np.full((height, width, 3), BG, np.uint8)
        x0, y0, rw, rh = 50, 40, width - 70, height - 150
        if self.cols:
            ras = diverging(self.H)
            img[y0:y0 + rh, x0:x0 + rw] = cv2.resize(ras, (rw, rh), interpolation=cv2.INTER_NEAREST) \
                if len(self.cols) == self.steps else _left_pad(ras, rw, rh, self.steps)
        labels = [(x0, 8, 'память M1: 512 ячеек GRU за последние 20 с (синий −, красный +)', INK)]
        n = len(self.cols)
        px = lambda i: x0 + int((self.steps - n + i + 0.5) * rw / self.steps)  # noqa: E731
        last_x = -10 ** 6  # events closer than a word apart keep their line but not their label
        for i, ev in enumerate(self.evs):
            if ev:
                cv2.line(img, (px(i), y0 - 4), (px(i), y0 + rh), INK, 1)
            if ev and px(i) - last_x >= 48:
                last_x = px(i)
                for k, e in enumerate(sorted(ev)):
                    labels.append((px(i) + 2, y0 - 18 - 12 * k, EVENT_TEXT[e], MUTED))
        zy0, zh = y0 + rh + 30, 60
        cv2.rectangle(img, (x0, zy0), (x0 + rw, zy0 + zh), (40, 40, 40), 1)
        pts = [(px(i), int(zy0 + zh - z * zh)) for i, z in enumerate(self.zs)]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, tuple(int(c) for c in BLUE), 2, cv2.LINE_AA)
        labels += [(x0, zy0 - 16, 'вентиль обновления (среднее)', MUTED), (4, zy0 - 6, 'держит', MUTED),
                   (4, zy0 + zh - 10, 'пишет', MUTED), (x0, height - 18, '−20 с', MUTED),
                   (x0 + rw - 30, height - 18, 'сейчас', MUTED)]
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for x, y, text, color in labels:
            d.text((x, y), text, font=F_SMALL, fill=(int(color[2]), int(color[1]), int(color[0])))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _left_pad(ras, rw, rh, steps):
    # fewer columns than the tape holds: the newest stay on the right
    full = np.zeros((ras.shape[0], steps, 3), np.uint8)
    full[:] = GRAY.astype(np.uint8)
    full[:, steps - ras.shape[1]:] = ras
    return cv2.resize(full, (rw, rh), interpolation=cv2.INTER_NEAREST)
