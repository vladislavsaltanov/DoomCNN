"""What happens inside a deathmatch network on one step, for the viewer.

trace() runs one forward pass with hooks: the maps of the three conv layers, the 512 features, the
GRU state and gates (M1), the logits of the four heads, the share of every unit and of every vector input
in every value of every head, and the saliency of the chosen move and turn over the input pixels. The model
gets the frames and the vector only. render() draws it as one 1280x480 picture.
"""
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from doomagent.brain import _font, _mosaic, saliency_overlay
from doomagent.dm.actions import HEAD_SIZES, MOVE_NAMES, TURN_DEG
from doomagent.dm.model import gru_gates, split_logits

W, H = 1280, 480
RU_MOVE = ('стоять', 'вперёд', 'назад', 'шаг влево', 'шаг вправо', 'вперёд-влево', 'вперёд-вправо', 'назад-влево',
           'назад-вправо')
RU_HEADS = ('движение', 'поворот', 'выстрел', 'оружие')
RU_VALUES = (RU_MOVE, tuple(f'{d:+g}°' for d in TURN_DEG), ('нет', 'огонь'), ('держать', 'следующее'))
RU_SLOTS = ('кулак', 'пистолет', 'дробовик', 'пулемёт', 'ракеты', 'плазма', 'BFG')
BG, INK, MUTED = (25, 26, 26), (236, 239, 240), (163, 149, 139)
POS, NEG, ACTIVE = (229, 135, 57), (103, 103, 230), (71, 181, 255)  # blue for, red against (BGR)
F, F_SMALL = _font(15), _font(12)


def _split(a):
    # rows of an (N_LOGITS, ...) array, per head
    return list(np.split(a, np.cumsum(HEAD_SIZES)[:-1]))


def trace(model, frames, vec, h=None):
    x = torch.from_numpy(np.ascontiguousarray(frames))[None].float().div(255.0).requires_grad_(True)
    v = torch.as_tensor(np.asarray(vec, np.float32))[None]
    layers = list(model.body)
    relus = [layers[i + 1] for i, m in enumerate(layers) if isinstance(m, torch.nn.Conv2d)]
    maps = []
    hooks = [r.register_forward_hook(lambda m, i, o: maps.append(o)) for r in relus]
    prev = (h[0] if h is not None else torch.zeros(1, 512)).detach().clone()

    def top(feat, vv):
        # everything above the convolutions: (head input, logits, gates)
        if model.memory:
            r, z, _, hid = gru_gates(model.gru, torch.cat([feat, vv], -1), prev)
            return hid, model.head(hid)[0], dict(r=r[0].detach().numpy(), z=z[0].detach().numpy())
        head_in = torch.cat([feat, vv], -1)
        return head_in, model.head(head_in)[0], None

    try:
        model.eval()
        feat = model.fc(model.body(x).flatten(1))
        head_in, logits, gates = top(feat, v)
        action = tuple(int(lg.argmax()) for lg in split_logits(logits))
        # saliency of the chosen move and turn
        off = np.cumsum((0,) + HEAD_SIZES)
        gx, = torch.autograd.grad(logits[off[0] + action[0]] + logits[off[1] + action[1]], x)
        saliency = gx[0].abs().amax(0).numpy()
    finally:
        for hk in hooks:
            hk.remove()
    # the vector's share in every logit (gradient x input) on the small graph above the frozen features
    vv = v.clone().requires_grad_(True)
    _, lg2, _ = top(feat.detach(), vv)
    vec_contrib = np.stack([(torch.autograd.grad(lg2[k], vv, retain_graph=True)[0][0] * v[0]).numpy()
                            for k in range(len(lg2))])
    w = model.head.weight.detach().numpy()
    unit = head_in[0, :512].detach().numpy()
    hid = head_in[0].detach().numpy() if model.memory else None
    lg = logits.detach()
    return dict(conv=[m[0].detach().numpy() for m in maps], feat=feat[0].detach().numpy(), hid=hid, gates=gates,
                logits=[p.numpy() for p in split_logits(lg)],
                probs=[torch.softmax(p, 0).numpy() for p in split_logits(lg)], action=action,
                contrib=_split(w[:, :512] * unit[None, :]), bias=_split(model.head.bias.detach().numpy()),
                vec_contrib=_split(vec_contrib), saliency=saliency,
                h_new=None if hid is None else head_in.detach()[None])


def render(t, frames, vec):
    img = np.full((H, W, 3), BG, np.uint8)
    labels = []
    small = [cv2.cvtColor(cv2.resize(f, (96, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
             for f in frames]
    for i, f in enumerate(small[:3]):
        img[40 + i * 76:112 + i * 76, 15:111] = f
    img[268:340, 15:111] = saliency_overlay(small[3], t['saliency'])
    labels += [(15, 12, 'вход', F, INK), (15, 345, 'что сильнее влияет', F_SMALL, MUTED),
               (15, 360, 'на движение и поворот', F_SMALL, MUTED)]
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
    # the vector and its share in the chosen move: health, armor, ammo, weapon (one-hot, summed), the
    # weapons owned with their ammo (summed)
    move = t['action'][0]
    c = t['vec_contrib'][0][move]
    hand = np.asarray(vec[3:10])
    slot = int(np.argmax(hand)) if hand.max() > 0 else -1
    groups = (('здоровье', vec[0], c[0]), ('броня', vec[1], c[1]), ('патроны', vec[2], c[2]),
              (RU_SLOTS[slot] if slot >= 0 else 'оружие', 1.0 if slot >= 0 else 0.0, float(c[3:10].sum())),
              ('запас оружия', float(np.mean(vec[10:16])), float(c[10:].sum())))
    for i, (name, val, cc) in enumerate(groups):
        y = 384 + 22 * i
        cv2.rectangle(img, (fx, y), (fx + int(70 * min(max(float(val), 0), 1)), y + 6), INK, -1)
        mid = fx + 150
        cv2.rectangle(img, (mid, y), (mid + int(np.clip(cc, -1, 1) * 25), y + 6), POS if cc >= 0 else NEG, -1)
        labels.append((fx, y - 14, f'{name} {float(val):.2f}  вклад в движение {cc:+.2f}', F_SMALL, INK))
    # four heads: the three most likely values each; lines from the strongest units into the chosen turn
    ox, y = 1040, 30
    turn_y = None
    for k, (name, probs) in enumerate(zip(RU_HEADS, t['probs'])):
        labels.append((ox, y, name, F, INK))
        y += 22
        for v in np.argsort(-probs)[:3 if len(probs) > 2 else 2]:
            chosen = v == t['action'][k]
            color = ACTIVE if chosen else (90, 80, 72)
            cv2.circle(img, (ox + 6, y + 8), 5, color, -1, cv2.LINE_AA)
            cv2.rectangle(img, (ox + 16, y + 12), (ox + 16 + int(110 * probs[v]), y + 15), color, -1)
            labels.append((ox + 16, y - 3, f'{RU_VALUES[k][v]}  {probs[v]:.2f}', F_SMALL, ACTIVE if chosen else INK))
            if chosen and k == 1:
                turn_y = y + 8
            y += 22
        y += 6
    contrib = t['contrib'][1][t['action'][1]]
    tops = np.argsort(-np.abs(contrib))[:12]
    cs = max(float(np.abs(contrib[tops]).max()), 1e-6)
    for u in tops:
        r, col = divmod(int(u), 16)
        p0 = (fx + col * cell + cell // 2, fy + r * cell + cell // 2)
        cv2.line(img, p0, (ox, turn_y), POS if contrib[u] > 0 else NEG, max(1, int(3 * abs(contrib[u]) / cs)),
                 cv2.LINE_AA)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for x, y, text, font, color in labels:
        d.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


assert len(RU_MOVE) == len(MOVE_NAMES)
