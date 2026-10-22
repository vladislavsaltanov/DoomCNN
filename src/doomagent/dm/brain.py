"""What happens inside a deathmatch network on one step, for the viewer.

trace() runs one forward pass with hooks: the maps of the three conv layers, the 512 features, the
GRU state and gates (M1), the logits, the share of every unit and of [health, rockets] in every
action, and the saliency of the chosen action over the input pixels. The model gets the frames and
the vector only. render() draws it as one 1280x480 picture.
"""
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from doomagent.brain import _font, _mosaic, saliency_overlay
from doomagent.dm.config import ACTION_NAMES
from doomagent.dm.model import gru_gates

W, H = 1280, 480
RU_ACTIONS = ('вперёд', 'вперёд-влево', 'вперёд-вправо', 'влево', 'вправо', 'чуть влево',
              'чуть вправо', 'шаг влево', 'шаг вправо', 'выстрел')
BG, INK, MUTED = (25, 26, 26), (236, 239, 240), (163, 149, 139)
POS, NEG, ACTIVE = (229, 135, 57), (103, 103, 230), (71, 181, 255)  # blue for, red against (BGR)
F, F_SMALL = _font(15), _font(12)


def trace(model, frames, vec, h=None):
    x = torch.from_numpy(np.ascontiguousarray(frames))[None].float().div(255.0).requires_grad_(True)
    v = torch.as_tensor(np.asarray(vec, np.float32))[None].requires_grad_(True)
    layers = list(model.body)
    relus = [layers[i + 1] for i, m in enumerate(layers) if isinstance(m, torch.nn.Conv2d)]
    maps = []
    hooks = [r.register_forward_hook(lambda m, i, o: maps.append(o)) for r in relus]
    try:
        model.eval()
        feat = model.fc(model.body(x).flatten(1))
        gates, hid = None, None
        if model.memory:
            prev = (h[0] if h is not None else torch.zeros(1, 512)).detach().clone()
            r, z, _, hid = gru_gates(model.gru, torch.cat([feat, v], -1), prev)
            head_in = hid
            gates = dict(r=r[0].detach().numpy(), z=z[0].detach().numpy())
        else:
            head_in = torch.cat([feat, v], -1)
        logits = model.head(head_in)[0]
        a = int(logits.argmax())
        vec_contrib = np.zeros((len(logits), 2), np.float32)
        for k in range(len(logits)):
            gx, gv = torch.autograd.grad(logits[k], [x, v], retain_graph=True)
            vec_contrib[k] = (gv[0] * v[0]).detach().numpy()
            if k == a:
                saliency = gx[0].abs().amax(0).numpy()
    finally:
        for hk in hooks:
            hk.remove()
    w = model.head.weight.detach().numpy()
    unit = head_in[0, :512].detach().numpy()
    return dict(conv=[m[0].detach().numpy() for m in maps], feat=feat[0].detach().numpy(),
                hid=None if hid is None else hid[0].detach().numpy(), gates=gates,
                logits=logits.detach().numpy(), probs=torch.softmax(logits.detach(), 0).numpy(),
                action=a, contrib=w[:, :512] * unit[None, :], bias=model.head.bias.detach().numpy(),
                vec_contrib=vec_contrib, saliency=saliency,
                h_new=None if hid is None else hid.detach()[None])


def render(t, frames, vec):
    img = np.full((H, W, 3), BG, np.uint8)
    labels = []
    small = [cv2.cvtColor(cv2.resize(f, (96, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
             for f in frames]
    for i, f in enumerate(small[:3]):
        img[40 + i * 76:112 + i * 76, 15:111] = f
    img[268:340, 15:111] = saliency_overlay(small[3], t['saliency'])
    labels += [(15, 12, 'вход', F, INK), (15, 345, 'что сильнее влияет', F_SMALL, MUTED),
               (15, 360, 'на решение', F_SMALL, MUTED)]
    blocks = [(t['conv'][0], 4, 50, 37, 128, 'conv1 32 карты'), (t['conv'][1], 8, 28, 21, 352, 'conv2 64'),
              (t['conv'][2], 8, 28, 21, 608, 'conv3 64')]
    for maps, cols, tw, th, x, name in blocks:
        m = _mosaic(maps, cols, tw, th, float(maps.max()))
        m = m[:H - 50, :W - x]
        img[40:40 + m.shape[0], x:x + m.shape[1]] = m
        labels.append((x, 12, name, F, INK))
    # 512 units: the fc features (M0) or the GRU state (M1), 16 x 32
    fx, fy, cell = 866, 40, 9
    unit = t['hid'] if t['hid'] is not None else t['feat']
    scale = max(float(np.abs(unit).max()), 1e-6)
    grid = (np.clip(np.abs(unit) / scale, 0, 1) * 255).astype(np.uint8).reshape(32, 16)
    img[fy:fy + 32 * cell, fx:fx + 16 * cell] = cv2.applyColorMap(
        cv2.resize(grid, (16 * cell, 32 * cell), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_INFERNO)
    labels.append((fx, 12, 'память GRU 512' if t['hid'] is not None else 'признаки fc 512', F, INK))
    if t['gates'] is not None:
        zbar = int(140 * float(t['gates']['z'].mean()))
        cv2.rectangle(img, (fx, fy + 32 * cell + 6), (fx + 140, fy + 32 * cell + 14), (60, 60, 60), -1)
        cv2.rectangle(img, (fx, fy + 32 * cell + 6), (fx + zbar, fy + 32 * cell + 14), POS, -1)
        labels.append((fx, fy + 32 * cell + 16, f'держит память: {t["gates"]["z"].mean():.2f}', F_SMALL, MUTED))
    # [health, rockets] and their share in the chosen action
    a = t['action']
    for i, (name, val) in enumerate((('здоровье', vec[0]), ('ракеты', vec[1]))):
        y = 400 + 34 * i
        c = float(t['vec_contrib'][a, i])
        cv2.rectangle(img, (fx, y), (fx + int(100 * min(max(val, 0), 1)), y + 8), INK, -1)
        mid = fx + 220
        cv2.rectangle(img, (mid, y), (mid + int(np.clip(c, -1, 1) * 60), y + 8), POS if c >= 0 else NEG, -1)
        labels.append((fx, y - 16, f'{name} {val:.2f}   вклад {c:+.2f}', F_SMALL, INK))
    # outputs and the strongest units into the chosen action
    ox, oy0, step = 1045, 52, 40
    contrib = t['contrib'][a]
    top = np.argsort(-np.abs(contrib))[:14]
    cs = max(float(np.abs(contrib[top]).max()), 1e-6)
    for u in top:
        r, col = divmod(int(u), 16)
        p0 = (fx + col * cell + cell // 2, fy + r * cell + cell // 2)
        cv2.line(img, p0, (ox - 8, oy0 + a * step), POS if contrib[u] > 0 else NEG,
                 max(1, int(4 * abs(contrib[u]) / cs)), cv2.LINE_AA)
    for i, name in enumerate(RU_ACTIONS):
        y = oy0 + i * step
        color = ACTIVE if i == a else (90, 80, 72)
        cv2.circle(img, (ox, y), 8, color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (ox + 16, y + 8), (ox + 16 + int(110 * t['probs'][i]), y + 13), color, -1)
        labels.append((ox + 16, y - 12, f'{name}  {t["probs"][i]:.2f}', F_SMALL, ACTIVE if i == a else INK))
    labels += [(ox - 20, 12, 'действие', F, INK)]
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for x, y, text, font, color in labels:
        d.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


assert len(RU_ACTIONS) == len(ACTION_NAMES)
