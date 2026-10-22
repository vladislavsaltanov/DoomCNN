import cv2
import numpy as np

from doomagent.dm.config import ACTION_NAMES, BARREL, BOT, DM, ITEM_KIND

# BGR box colors by label class (palette slots: red, aqua, yellow); for viewers only
BOX = {'bot': (72, 73, 227), 'item': (122, 175, 27), 'barrel': (0, 161, 237)}
INK, MUTED, ACTIVE = (236, 239, 240), (163, 149, 139), (71, 181, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label_class(name):
    if name == BOT:
        return 'bot'
    if name in ITEM_KIND:
        return 'item'
    return 'barrel' if name == BARREL else None


def compose_view(rgb, priv, action=None, info=None, cfg=DM, boxes=True, hud=None):
    # the game frame for people: label boxes, crosshair, the chosen action (and its probabilities),
    # match counters, the Doom status bar; the agent never sees this image
    img = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    sx, sy = W / cfg.width, H / cfg.height
    if boxes:
        for lb in priv.labels:
            k = label_class(lb.object_name)
            if k:
                cv2.rectangle(img, (int(lb.x * sx), int(lb.y * sy)),
                              (int((lb.x + lb.width) * sx), int((lb.y + lb.height) * sy)), BOX[k],
                              2 if k == 'bot' else 1)
    cx, cy = W // 2, H // 2
    cv2.line(img, (cx, cy - 8), (cx, cy + 8), INK, 1)
    cv2.line(img, (cx - 8, cy), (cx + 8, cy), INK, 1)
    if action is not None:
        probs = None if info is None else info.get('probs')
        for i, name in enumerate(ACTION_NAMES):
            y = 18 + 17 * i
            color = ACTIVE if i == action else INK
            if probs is not None:
                cv2.rectangle(img, (112, y - 10), (112 + int(110 * probs[i]), y + 1), color, -1)
            cv2.putText(img, name + ('' if probs is not None or i != action else ' <'), (8, y), FONT,
                        0.42, color, 1, cv2.LINE_AA)
    text = f'frags {priv.frags:.0f}  deaths {priv.deaths:.0f}'
    (tw, _), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    cv2.putText(img, text, (W - tw - 10, 20), FONT, 0.5, INK, 1, cv2.LINE_AA)
    if hud is not None:
        img = hud.draw(img, priv.health, priv.ammo, priv.armor)
    return img
