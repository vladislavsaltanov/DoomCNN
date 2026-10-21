"""The Doom status bar, drawn for the viewer only.

The network was trained on frames without the HUD (Cfg.hud = False), so the game keeps rendering
without it; this module paints the real status bar graphics from freedoom2.wad (STBAR, the big
red digits, the face) over the bottom of the viewer's frame.
"""
import os
import struct

import cv2
import numpy as np

from doomagent.wadgen import read_wad

# positions from Doom's st_stuff.c, in 320x200 screen units relative to the bar (y = 168)
AMMO_X, HEALTH_X, ARMOR_X, NUM_Y = 44, 90, 221, 3
FACE_X, FACE_Y = 143, 0


def decode_patch(b, pal):
    # Doom picture format -> BGRA image and its (left, top) offsets
    w, h, left, top = struct.unpack_from('<hhhh', b, 0)
    img = np.zeros((h, w, 4), np.uint8)
    for x in range(w):
        p = struct.unpack_from('<I', b, 8 + 4 * x)[0]
        while b[p] != 255:
            y0, n = b[p], b[p + 1]
            img[y0:y0 + n, x, :3] = pal[list(b[p + 3:p + 3 + n])][:, ::-1]
            img[y0:y0 + n, x, 3] = 255
            p += n + 4
    return img, left, top


def _paste(dst, patch, x, y):
    img, left, top = patch
    x, y = x - left, y - top
    h, w = img.shape[:2]
    y0, x0 = max(y, 0), max(x, 0)
    y1, x1 = min(y + h, dst.shape[0]), min(x + w, dst.shape[1])
    if y1 <= y0 or x1 <= x0:
        return
    src = img[y0 - y:y1 - y, x0 - x:x1 - x]
    a = src[..., 3:] > 0
    dst[y0:y1, x0:x1] = np.where(a, src[..., :3], dst[y0:y1, x0:x1])


class Hud:
    def __init__(self, wad=None):
        if wad is None:
            import vizdoom as vzd
            wad = os.path.join(os.path.dirname(vzd.__file__), 'freedoom2.wad')
        lumps = dict(read_wad(wad))
        pal = np.frombuffer(lumps['PLAYPAL'][:768], np.uint8).reshape(256, 3)
        names = ['STBAR', 'STARMS', 'STTPRCNT', 'STFDEAD0'] + [f'STTNUM{i}' for i in range(10)]
        names += [f'STF{k}{p}' for p in range(5) for k in ('EVL', 'OUCH')]
        names += [f'STFST{p}{i}' for p in range(5) for i in range(3)]
        self.p = {n: decode_patch(lumps[n], pal) for n in names if n in lumps}

    def _number(self, bar, value, right):
        x = right
        for d in reversed(str(max(int(value), 0))):
            img = self.p[f'STTNUM{d}']
            x -= img[0].shape[1]
            _paste(bar, img, x, NUM_Y)

    def bar(self, health, ammo, armor=0, killed=False, hurt=False):
        # one 320x32 status bar
        bar = np.zeros((32, 320, 3), np.uint8)
        _paste(bar, self.p['STBAR'], 0, 0)
        _paste(bar, self.p['STARMS'], 104, 0)
        self._number(bar, ammo, AMMO_X)
        self._number(bar, health, HEALTH_X)
        _paste(bar, self.p['STTPRCNT'], HEALTH_X, NUM_Y)
        self._number(bar, armor, ARMOR_X)
        _paste(bar, self.p['STTPRCNT'], ARMOR_X, NUM_Y)
        pain = min(4, int((100 - min(health, 100)) * 5 / 101))
        if health <= 0:
            face = 'STFDEAD0'
        elif hurt:
            face = f'STFOUCH{pain}'
        elif killed:
            face = f'STFEVL{pain}'
        else:
            face = f'STFST{pain}1'
        _paste(bar, self.p[face], FACE_X, FACE_Y)
        return bar

    def draw(self, bgr, health, ammo, armor=0, killed=False, hurt=False):
        # the bar scaled to the frame width, over the bottom of a copy of the frame
        out = np.ascontiguousarray(bgr).copy()
        w = out.shape[1]
        bar = cv2.resize(self.bar(health, ammo, armor, killed, hurt), (w, w * 32 // 320),
                         interpolation=cv2.INTER_NEAREST)
        out[-bar.shape[0]:] = bar
        return out
