"""Demo videos for people: the game, what the network sees and decides, and live charts.

    python scripts/demo_video.py --model runs/ppo_ft_flip/best_model.zip --variants base tex_val_00 \
        --seed 6000 --out artifacts/demo.mp4
    python scripts/demo_video.py --model runs/r4.pt --compare runs/ppo_ft_flip/best_model.zip \
        --variants mix_val_01 --seed 6000 --out artifacts/compare.mp4

1280x720, 35 frames per second (one per game tic), H.264. The labels, boxes and teacher action
are for the viewer only; the network gets the 4 grey frames shown on the right.
"""
import argparse
import os
import subprocess
import tempfile
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from doomagent.config import ACTION_NAMES, Cfg
from doomagent.env import DoomEnv
from doomagent.hud import Hud
from doomagent.model import StudentCNN
from doomagent.teacher import Teacher, enemies_from_labels, is_enemy_name, nearest_aux
from doomagent.variants import get_variant
from doomagent.viz import VideoSink, h264_args

W, H = 1280, 720
RU = {'turn_left': 'влево', 'turn_right': 'вправо', 'shoot': 'выстрел', 'fine_left': 'чуть влево',
      'fine_right': 'чуть вправо'}
SPLIT = {'base': 'базовая графика', 'train': 'графика из обучения', 'val': 'незнакомая графика (val)',
         'test': 'незнакомая графика (test)'}
BG, PANEL, INK, MUTED = (19, 16, 14), (31, 26, 22), (238, 233, 230), (163, 149, 139)
ORANGE, GREEN, RED, BLUE = (71, 181, 255), (139, 209, 79), (107, 107, 255), (255, 169, 90)


def font(size, bold=False):
    name = 'segoeuib.ttf' if bold else 'segoeui.ttf'
    for p in (os.path.join(os.environ.get('WINDIR', 'C:/Windows'), 'Fonts', name),):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_TITLE, F_BIG, F, F_SMALL = font(26, True), font(22, True), font(17), font(14)
HUD = Hud()


def load_model(path, cfg):
    # a .pt student, or a PPO .zip whose actor is copied into a StudentCNN; the auxiliary heads
    # (enemy, offset, mask) then come from r4, which PPO started from and never trained
    m = StudentCNN(cfg)
    if path.endswith('.zip'):
        from stable_baselines3 import PPO
        from doomagent.ppo import policy_state
        base = torch.load('runs/r4.pt', map_location='cpu')
        m.load_state_dict(policy_state(PPO.load(path, device='cpu'), base))
    else:
        m.load_state_dict(torch.load(path, map_location='cpu'))
    m.ppo = path.endswith('.zip')
    return m.eval()


def chart(img, x, y, w, h, series, lo, hi, n=150):
    # series: [(values, BGR, dashed)]; the last n steps, newest on the right
    cv2.rectangle(img, (x, y), (x + w, y + h), (45, 39, 34), -1)
    for frac in (0.0, 0.5, 1.0):
        yy = int(y + h - frac * h)
        cv2.line(img, (x, yy), (x + w, yy), (60, 53, 47), 1)
    for vals, color, dashed in series:
        vals = list(vals)[-n:]
        pts = [(x + int(i / max(n - 1, 1) * w), int(y + h - (min(max(v, lo), hi) - lo) / (hi - lo) * h))
               if v is not None else None for i, v in enumerate(vals)]
        for i, (a, b) in enumerate(zip(pts, pts[1:])):
            if a is None or b is None or (dashed and i % 4 >= 2):
                continue
            cv2.line(img, a, b, color, 2, cv2.LINE_AA)


def compose(rgb, obs, info, st, hist, cfg):
    img = np.full((H, W, 3), BG, np.uint8)
    # the game, with what the network sees drawn over it
    game = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    gh, gw = game.shape[:2]
    if info.get('seg') is not None:
        heat = cv2.resize(info['seg'].astype(np.float32), (gw, gh))
        a = (np.clip(heat, 0, 1) * 0.5)[..., None]
        game = (game * (1 - a) + np.array([0, 0, 255]) * a).astype(np.uint8)
    sx, sy = gw / cfg.width, gh / cfg.height
    for lb in st['boxes']:
        cv2.rectangle(game, (int(lb.x * sx), int(lb.y * sy)),
                      (int((lb.x + lb.width) * sx), int((lb.y + lb.height) * sy)), GREEN, 1)
    if info.get('vis', 0) > 0.5 and not st['ppo']:  # PPO never trained the offset head
        xo = int((info['off'] + 1) / 2 * gw)
        cv2.line(game, (xo, 0), (xo, gh), ORANGE, 2)
    cv2.line(game, (gw // 2, gh // 2 - 9), (gw // 2, gh // 2 + 9), INK, 1)
    cv2.line(game, (gw // 2 - 9, gh // 2), (gw // 2 + 9, gh // 2), INK, 1)
    game = HUD.draw(game, st['health'], st['ammo'], killed=st.get('killed', False),
                    hurt=st.get('hurt', False))  # viewer only; the network never sees it
    img[60:60 + 645, 15:15 + 860] = cv2.resize(game, (860, 645), interpolation=cv2.INTER_LINEAR)
    # right column
    x0 = 895
    cv2.rectangle(img, (x0 - 5, 60), (W - 12, H - 12), PANEL, -1)
    probs = info['probs']
    for i, name in enumerate(ACTION_NAMES):
        y = 150 + i * 30
        cv2.rectangle(img, (x0 + 110, y), (x0 + 300, y + 14), (45, 39, 34), -1)
        col = ORANGE if i == st['action'] else BLUE
        cv2.rectangle(img, (x0 + 110, y), (x0 + 110 + int(190 * probs[i]), y + 14), col, -1)
    tiles = [cv2.cvtColor(cv2.resize(f, (68, 51), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
             for f in obs]
    seg = cv2.resize((np.clip(info.get('seg', np.zeros((2, 2))), 0, 1) * 255).astype(np.uint8),
                     (68, 51), interpolation=cv2.INTER_NEAREST)
    tiles.append(cv2.applyColorMap(seg, cv2.COLORMAP_INFERNO))
    for i, t in enumerate(tiles):
        img[340:391, x0 + i * 73:x0 + i * 73 + 68] = t
    chart(img, x0, 445, 365, 110, [(hist['agree'], GREEN, False), (hist['vis_p'], ORANGE, False),
                                   (hist['vis_t'], RED, True)], 0.0, 1.0)
    chart(img, x0, 595, 365, 80, [(hist['off_p'], ORANGE, False)] * (not st['ppo'])
          + [(hist['off_t'], RED, True)], -1.0, 1.0)
    # text (Cyrillic needs a TrueType font)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    rgb_ = lambda c: (c[2], c[1], c[0])  # noqa: E731
    d.text((15, 14), st['title'], font=F_TITLE, fill=rgb_(INK))
    d.text((x0, 70), f"награда {st['reward']:.0f}   убито {st['kills']:.0f}   шаг {st['step']}",
           font=F_BIG, fill=rgb_(INK))
    d.text((x0, 100), f"патроны {st['ammo']:.0f}   здоровье {st['health']:.0f}   "
                      f"как учитель {st['agree']:.0%}", font=F, fill=rgb_(MUTED))
    for i, name in enumerate(ACTION_NAMES):
        y = 145 + i * 30
        c = ORANGE if i == st['action'] else MUTED
        d.text((x0, y), RU[name] + (' ●' if i == st['teacher'] else ''), font=F, fill=rgb_(c))
        d.text((x0 + 308, y), f'{probs[i]:.2f}', font=F_SMALL, fill=rgb_(MUTED))
    d.text((x0, 305), f"враг в кадре: {info.get('vis', 0):.0%}" + (' (головы r4)' if st['ppo'] else '')
           + '  ·  вход сети и карта врагов',
           font=F_SMALL, fill=rgb_(MUTED))
    d.text((x0, 405), 'совпадение с учителем · враг: сеть · враг: правда', font=F_SMALL,
           fill=rgb_(MUTED))
    d.text((x0, 422), '(зелёный)              (оранжевый)  (красный пунктир)', font=F_SMALL,
           fill=rgb_(MUTED))
    d.text((x0, 565), 'направление на врага: правда' + (' (у PPO голова не обучалась)' if st['ppo']
           else ' и сеть') + ' (−1 … +1)', font=F_SMALL, fill=rgb_(MUTED))
    d.text((x0, 682), 'оранжевое — выбор сети, ● — что сделал бы учитель', font=F_SMALL,
           fill=rgb_(MUTED))
    if st.get('banner'):
        d.rectangle((15, 300, 875, 400), fill=(14, 16, 19))
        d.text((35, 318), st['banner'][0], font=F_TITLE, fill=rgb_(ORANGE))
        d.text((35, 358), st['banner'][1], font=F, fill=rgb_(INK))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def play(model, label, variant_name, seed, sink, cfg):
    v = get_variant(variant_name)
    env, teacher = DoomEnv(cfg, v), Teacher(cfg)
    obs, done, step = env.reset(seed), False, 0
    teacher.reset()
    st = dict(reward=0.0, kills=0.0, step=0, agree=0.0, ammo=26, health=100, boxes=[], ppo=model.ppo,
              title=f'{label}  ·  {variant_name} — {SPLIT.get(v.split, v.split)}  ·  seed {seed}')
    hist = {k: deque(maxlen=150) for k in ('agree', 'vis_p', 'vis_t', 'off_p', 'off_t')}
    agree, raw = 0, deque(maxlen=20)
    banner_tics = 70
    while not done:
        priv = env.privileged()
        info = model.explain(obs)
        action, label_t = info['action'], teacher(obs, priv)
        agree += action == label_t
        vis_t, off_t = nearest_aux(enemies_from_labels(priv.labels, priv.px, priv.py, cfg.width))
        raw.append(float(action == label_t))
        hist['agree'].append(sum(raw) / len(raw))  # agreement over the last 20 steps
        hist['vis_p'].append(info['vis'])
        hist['vis_t'].append(float(vis_t))
        hist['off_p'].append(info['off'] if info['vis'] > 0.5 else None)
        hist['off_t'].append(off_t if vis_t else None)
        st.update(action=action, teacher=label_t, step=step, ammo=priv.ammo, health=priv.health,
                  agree=agree / (step + 1),
                  boxes=[lb for lb in priv.labels if is_enemy_name(lb.object_name)])

        def tic(rgb, p):
            nonlocal banner_tics
            st['banner'] = ((st['title'].split('  ·  ')[1], 'сеть видит только серые кадры справа')
                            if banner_tics > 0 else None)
            banner_tics -= 1
            sink.write(compose(rgb, obs, info, st, hist, cfg))

        obs, r, done = env.step(action, on_tic=tic)
        st['reward'] += r
        st['kills'] += max(r, 0)
        step += 1
    env.close()
    return st['reward'], step


def duration(path):
    return float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                 '-of', 'csv=p=0', path], capture_output=True, text=True).stdout)


def side_by_side(left, right, out):
    # 960x540 each; the shorter video holds its last frame exactly until the longer one ends
    dl, dr = duration(left), duration(right)
    pad = lambda d: f',tpad=stop_mode=clone:stop_duration={max(dl, dr) - d:.3f}'  # noqa: E731
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', left, '-i', right, '-filter_complex',
                    f'[0:v]scale=960:540{pad(dl)}[a];[1:v]scale=960:540{pad(dr)}[b];'
                    '[a][b]hstack=inputs=2[v]', '-map', '[v]', *h264_args(), out], check=True)


def record(model_path, label, variants, seed, out, cfg):
    model = load_model(model_path, cfg)
    sink = VideoSink(out, 35, (W, H))
    for name in variants:
        r, n = play(model, label, name, seed, sink, cfg)
        print(f'{label} {name} seed={seed}: reward {r:.0f}, {n} steps', flush=True)
    sink.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--label')
    ap.add_argument('--compare', help='a second model, played on the same seeds, shown side by side')
    ap.add_argument('--compare-label')
    ap.add_argument('--variants', nargs='+', default=['base'])
    ap.add_argument('--seed', type=int, default=6000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    cfg = Cfg()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    name = lambda p: p.replace('runs/', '').replace('/best_model.zip', '')  # noqa: E731
    if not a.compare:
        record(a.model, a.label or name(a.model), a.variants, a.seed, a.out, cfg)
    else:
        tmp = tempfile.mkdtemp()
        left, right = os.path.join(tmp, 'a.mp4'), os.path.join(tmp, 'b.mp4')
        record(a.model, a.label or name(a.model), a.variants, a.seed, left, cfg)
        record(a.compare, a.compare_label or name(a.compare), a.variants, a.seed, right, cfg)
        side_by_side(left, right, a.out)
    print('written', a.out)
