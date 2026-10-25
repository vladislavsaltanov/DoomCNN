"""Demo video: the teacher, then each model, on the first seconds of the same match.

    python scripts/dm/dm_demo_video.py --models runs/dm/m0_pilot.pt runs/dm/m1_pilot.pt --seed 35000 \
        --seconds 40 --out artifacts/dm/v1/d2_pilot/d2_demo.mp4 --brain-out artifacts/dm/v1/d2_pilot/d2_brain_m1.mp4

1280x720 at 35 frames per second: a 1.5 s title, then the game (boxes, status bar) on the left; on
the right the minimap, the ten action probabilities (the teacher's mode for the teacher), the
[health, rockets] bars, the M1 memory over the last 5 s, and running charts of the agreement with the
shadow teacher and of the frags. --brain-out writes the M1 network view over its memory tape,
1280x900, one picture per agent step held for its four tics.
"""
import argparse
import os

import cv2
import numpy as np

from doomagent.dm.actions import HEADS, MOVE_NAMES, TURN_DEG
from doomagent.dm.config import DM

VALUES = (MOVE_NAMES, tuple(f'{d:+g}' for d in TURN_DEG), ('no', 'fire'), ('keep', 'next'))
from doomagent.dm.env import DeathmatchEnv
from doomagent.dm.memory_view import MemoryTape, diverging, events_from
from doomagent.dm.minimap import MapView
from doomagent.dm.teacher import DMTeacher
from doomagent.dm.viewer import compose_view
from doomagent.hud import Hud
from doomagent.viz import VideoSink

W, H, FPS = 1280, 720, 35
BG, INK, MUTED, ACTIVE = (25, 26, 26), (236, 239, 240), (163, 149, 139), (71, 181, 255)
SERIES = (229, 135, 57)  # BGR of the dashboard blue
FONT = cv2.FONT_HERSHEY_SIMPLEX
TAPE_STEPS = 44  # 5 s of agent steps


def text(img, s, x, y, scale=0.45, color=INK, thick=1):
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def title(name, sub):
    img = np.full((H, W, 3), BG, np.uint8)
    text(img, name, 80, 330, 1.6, INK, 2)
    text(img, sub, 80, 390, 0.8, MUTED, 1)
    return img


def run_chart(img, x, y, w, h, ys, lo, hi, name, value):
    cv2.rectangle(img, (x, y), (x + w, y + h), (45, 45, 45), 1)
    text(img, name, x, y - 6, 0.42, MUTED)
    if len(ys) > 1:
        n = max(len(ys), 2)
        pts = np.array([(x + int(i * w / (n - 1)), y + h - int((v - lo) / ((hi - lo) or 1) * h))
                        for i, v in enumerate(ys)], np.int32)
        cv2.polylines(img, [pts], False, SERIES, 2, cv2.LINE_AA)
        text(img, value, min(pts[-1][0] + 4, x + w - 40), max(pts[-1][1] - 6, y + 12), 0.42, INK)


def side_panel(p, teacher, mapview, info, label, tape, agree, frags):
    img = np.full((H, W - 860, 3), BG, np.uint8)
    img[10:270, 80:340] = cv2.resize(mapview.draw(p, teacher), (260, 260), interpolation=cv2.INTER_AREA)
    y0 = 300
    text(img, f'teacher mode: {teacher.mode}', 16, y0 - 8, 0.45, MUTED)
    for k, h in enumerate(HEADS):
        y = y0 + 16 + 30 * k
        chosen = int(np.argmax(info['probs'][k])) if info is not None else int(label[k])
        text(img, h, 16, y, 0.42, MUTED)
        text(img, VALUES[k][chosen] + (f'  {info["probs"][k][chosen]:.2f}' if info is not None else ''), 90, y, 0.45,
             ACTIVE)
        if info is not None:
            cv2.rectangle(img, (90, y + 4), (90 + int(140 * info['probs'][k][chosen]), y + 8), ACTIVE, -1)
            text(img, f'teacher: {VALUES[k][int(label[k])]}', 250, y, 0.4, INK if chosen == int(label[k]) else MUTED)
    y = 460
    for i, (name, v, scale) in enumerate((('health', p.health, 100.0), ('ammo', p.weapon_ammo, 200.0))):
        cv2.rectangle(img, (130, y + 22 * i - 9), (130 + int(140 * min(max(v, 0) / scale, 1)), y + 22 * i + 1), INK, -1)
        text(img, f'{name} {max(v, 0):.0f}', 16, y + 22 * i, 0.42, INK)
    if tape is not None and tape.cols:
        ras = diverging(np.stack(list(tape.cols)[-TAPE_STEPS:], 1))
        full = np.full((ras.shape[0], TAPE_STEPS, 3), (56, 56, 53), np.uint8)
        full[:, TAPE_STEPS - ras.shape[1]:] = ras
        img[525:585, 16:404] = cv2.resize(full, (388, 60), interpolation=cv2.INTER_AREA)
        text(img, 'M1 memory, last 5 s (512 GRU units)', 16, 518, 0.4, MUTED)
    run_chart(img, 16, 625, 180, 80, agree, 0, 1, 'move+turn = teacher',
              f'{agree[-1]:.0%}' if agree else '')
    run_chart(img, 224, 625, 180, 80, frags, min(min(frags, default=0), 0), max(max(frags, default=1), 1),
              'frags', f'{frags[-1]:.0f}' if frags else '')
    return img


def play(name, path, seed, seconds, sink, brain_sink):
    agent = None
    if path is not None:
        from doomagent.dm.agent import DMAgent
        from doomagent.dm.model import load_model
        agent = DMAgent(load_model(path, DM))
    env, teacher, hud = DeathmatchEnv(DM), DMTeacher(DM), Hud()
    obs = env.reset(seed)
    mapview = MapView(env.sectors)
    tape = MemoryTape() if agent is not None and agent.model.memory else None
    agree_hist, agree, frags, prev = [], [], [], None
    steps = seconds * FPS // DM.frame_skip
    for step in range(steps):
        priv = env.privileged()
        label = teacher(obs, priv)
        info = None if agent is None else agent.explain(obs.frames, obs.vec, obs.first)
        action = label if info is None else info['action']
        if tape is not None:
            tape.push(info['h'], info['z'], events_from(prev, priv, obs.first, step == 0))
        if brain_sink is not None and tape is not None:
            import torch
            from doomagent.dm.brain import render, trace
            h = None if info['h_prev'] is None else torch.from_numpy(info['h_prev']).reshape(1, 1, 512)
            pic = np.full((900, 1280, 3), BG, np.uint8)
            pic[:480] = render(trace(agent.model, obs.frames, obs.vec, h), obs.frames, obs.vec)
            pic[480:900, 190:1090] = tape.render()
            for _ in range(DM.frame_skip):
                brain_sink.write(pic)
        prev = priv
        if info is not None:
            agree_hist.append(int(tuple(action[:2]) == tuple(label[:2])))  # move and turn both
            agree.append(float(np.mean(agree_hist[-25:])))

        def tic(rgb, p, action=action, info=info, label=label):
            game = compose_view(rgb, p, action, info, DM, True, hud, None, teacher)
            frame = np.full((H, W, 3), BG, np.uint8)
            frame[:645, :860] = cv2.resize(game, (860, 645), interpolation=cv2.INTER_LINEAR)
            text(frame, f'{name}   seed {seed}   frags {p.frags:.0f}   deaths {p.deaths:.0f}', 16, 690, 0.65, INK)
            frame[:, 860:] = side_panel(p, teacher, mapview, info, label, tape, agree, frags)
            sink.write(frame)

        obs, _, done = env.step(action, on_tic=tic)
        frags.append(float(env.privileged().frags))
        if done:
            break
    p = env.privileged()
    env.close()
    print(f'{name}: frags {p.frags:.0f}, deaths {p.deaths:.0f} in {seconds} s'
          + (f', agreement {np.mean(agree_hist):.0%}' if agree_hist else ''))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--seed', type=int, default=35000)
    ap.add_argument('--seconds', type=int, default=40)
    ap.add_argument('--out', required=True)
    ap.add_argument('--brain-out')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    sink = VideoSink(a.out, FPS, (W, H))
    brain_sink = None
    try:
        players = [('teacher', None, 'privileged: map, positions, labels')]
        for m in a.models:
            from doomagent.dm.model import load_model
            kind = 'M1 (GRU memory)' if load_model(m, DM).memory else 'M0 (no memory)'
            players.append((os.path.basename(m).replace('.pt', ''), m, f'{kind}: frames + [health, rockets] only'))
        for name, path, sub in players:
            for _ in range(int(1.5 * FPS)):
                sink.write(title(name, sub))
            bs = None
            if a.brain_out and path is not None and load_model(path, DM).memory and brain_sink is None:
                brain_sink = bs = VideoSink(a.brain_out, FPS, (1280, 900))
            play(name, path, a.seed, a.seconds, sink, bs)
    finally:
        sink.close()
        if brain_sink is not None:
            brain_sink.close()
    print('written', a.out, a.brain_out or '')
