"""What happens inside the network on one step, for the viewer.

trace() runs one forward pass with hooks: the feature maps of the three conv layers, the 512
units of the fc layer, the logits, and the contribution of every fc unit to every action
(activation x weight; plus the bias they add up to the logit). saliency is the gradient of the
chosen action's logit with respect to the input pixels: where a change would move the decision.
render() draws all of it as one 1280x430 picture.
"""
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from doomagent.config import ACTION_NAMES

W, H = 1280, 430
RU = {'turn_left': 'влево', 'turn_right': 'вправо', 'shoot': 'выстрел', 'fine_left': 'чуть влево',
      'fine_right': 'чуть вправо'}
BG, INK, MUTED = (19, 16, 14), (238, 233, 230), (163, 149, 139)
POS, NEG, ACCENT = (139, 209, 79), (107, 107, 255), (71, 181, 255)


def trace(model, obs):
    x = torch.from_numpy(np.asarray(obs)).unsqueeze(0).float().div(255.0)
    x.requires_grad_(True)
    layers = list(model.body)
    relus = [layers[i + 1] for i, m in enumerate(layers) if isinstance(m, torch.nn.Conv2d)]
    maps = []
    hooks = [r.register_forward_hook(lambda m, i, o: maps.append(o)) for r in relus]
    try:
        model.eval()
        f = model.body(x)
        h = model.fc(f.flatten(1))
        logits = model.pi(h)
        a = int(logits.argmax(1))
        logits[0, a].backward()
    finally:
        for hk in hooks:
            hk.remove()
    fc = h[0].detach().numpy()
    w = model.pi.weight.detach().numpy()
    probs = torch.softmax(logits.detach(), 1)[0].numpy()
    return dict(conv=[m[0].detach().numpy() for m in maps], fc=fc, logits=logits[0].detach().numpy(),
                probs=probs, action=a, contrib=fc[None, :] * w,
                saliency=x.grad[0].abs().amax(0).numpy())


def _font(size):
    p = os.path.join(os.environ.get('WINDIR', 'C:/Windows'), 'Fonts', 'segoeui.ttf')
    return ImageFont.truetype(p, size) if os.path.exists(p) else ImageFont.load_default()


F, F_SMALL = _font(15), _font(12)


def _mosaic(maps, cols, tile_w, tile_h, vmax):
    # all feature maps of one layer in a grid; one brightness scale for the whole layer
    n = len(maps)
    rows = (n + cols - 1) // cols
    out = np.full((rows * (tile_h + 2), cols * (tile_w + 2), 3), 30, np.uint8)
    for i, m in enumerate(maps):
        v = (np.clip(m / max(vmax, 1e-6), 0, 1) * 255).astype(np.uint8)
        t = cv2.applyColorMap(cv2.resize(v, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST),
                              cv2.COLORMAP_INFERNO)
        r, c = divmod(i, cols)
        out[r * (tile_h + 2):r * (tile_h + 2) + tile_h, c * (tile_w + 2):c * (tile_w + 2) + tile_w] = t
    return out


def saliency_overlay(bgr, sal):
    # the input-pixel saliency, smoothed, over a viewer frame
    s = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 2.0)
    s = cv2.resize(s / max(float(s.max()), 1e-8), (bgr.shape[1], bgr.shape[0]))
    heat = cv2.applyColorMap((s * 255).astype(np.uint8), cv2.COLORMAP_JET)
    a = (s ** 0.7 * 0.6)[..., None]
    return (bgr * (1 - a) + heat * a).astype(np.uint8)


def render(t, obs):
    img = np.full((H, W, 3), BG, np.uint8)
    labels = []  # (x, y, text, font, color) drawn after the pixels

    # input: the 4 frames, the saliency over the newest one
    frames = [cv2.cvtColor(cv2.resize(f, (96, 72)), cv2.COLOR_GRAY2BGR) for f in obs]
    for i, f in enumerate(frames[:3]):
        img[40 + i * 76:40 + i * 76 + 72, 15:111] = f
    img[40 + 3 * 76:40 + 3 * 76 + 72, 15:111] = saliency_overlay(frames[3], t['saliency'])
    labels += [(15, 12, 'вход', F, INK),
               (15, 345, 'что сильнее', F_SMALL, MUTED),
               (15, 360, 'влияет на', F_SMALL, MUTED),
               (15, 375, 'решение', F_SMALL, MUTED)]

    # conv layers
    blocks = [(t['conv'][0], 4, 50, 37, 128, 'conv1 32×29×39', 'ядро 8×8, шаг 4'),
              (t['conv'][1], 8, 28, 21, 352, 'conv2 64×13×18', 'ядро 4×4, шаг 2'),
              (t['conv'][2], 8, 28, 21, 608, 'conv3 64×11×16', 'ядро 3×3, шаг 1')]
    for maps, cols, tw, th, x, name, kern in blocks:
        m = _mosaic(maps, cols, tw, th, float(maps.max()))
        img[40:40 + m.shape[0], x:x + m.shape[1]] = m
        labels += [(x, 12, name, F, INK), (x, 40 + m.shape[0] + 4, kern, F_SMALL, MUTED)]

    # fc: 512 units as a 16x32 grid
    fx, fy, cell = 866, 40, 9
    fc = t['fc'] / max(float(t['fc'].max()), 1e-6)
    grid = (np.clip(fc, 0, 1) * 255).astype(np.uint8).reshape(32, 16)
    g = cv2.applyColorMap(cv2.resize(grid, (16 * cell, 32 * cell), interpolation=cv2.INTER_NEAREST),
                          cv2.COLORMAP_INFERNO)
    img[fy:fy + 32 * cell, fx:fx + 16 * cell] = g
    labels += [(fx, 12, 'fc 512', F, INK),
               (fx, fy + 32 * cell + 4, f'активно {int((t["fc"] > 0).sum())} из 512', F_SMALL, MUTED)]

    # outputs and the strongest connections into the chosen action
    ox, oy0, step = 1045, 70, 62
    a = t['action']
    c = t['contrib'][a]
    top = np.argsort(-np.abs(c))[:14]
    scale = max(float(np.abs(c[top]).max()), 1e-6)
    for u in top:
        r, col = divmod(int(u), 16)
        p0 = (fx + col * cell + cell // 2, fy + r * cell + cell // 2)
        p1 = (ox - 8, oy0 + a * step)
        cv2.line(img, p0, p1, POS if c[u] > 0 else NEG, max(1, int(4 * abs(c[u]) / scale)),
                 cv2.LINE_AA)
        cv2.rectangle(img, (p0[0] - 4, p0[1] - 4), (p0[0] + 4, p0[1] + 4), INK, 1)
    for i, name in enumerate(ACTION_NAMES):
        y = oy0 + i * step
        color = ACCENT if i == a else (90, 80, 72)
        cv2.circle(img, (ox, y), 9, color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (ox + 16, y + 10), (ox + 16 + int(110 * t['probs'][i]), y + 16), color, -1)
        labels.append((ox + 16, y - 12, f'{RU[name]}  {t["probs"][i]:.2f}', F,
                       ACCENT if i == a else INK))
    labels += [(ox - 20, 12, 'выход: действие', F, INK),
               (ox - 20, 375, 'линии — самые сильные связи fc → действие:', F_SMALL, MUTED),
               (ox - 20, 390, 'зелёные толкают к нему, красные против', F_SMALL, MUTED)]

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for x, y, text, font, color in labels:
        d.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
