import time

import cv2
import numpy as np

from doomagent.config import ACTION_NAMES, Cfg

FONT = cv2.FONT_HERSHEY_SIMPLEX


def stack_image(obs, path, scale=2):
    row = np.concatenate(list(obs), axis=1)
    row = cv2.resize(row, (row.shape[1] * scale, row.shape[0] * scale),
                     interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, row)


def draw_overlay(rgb, action, info=None, boxes=(), net_frame=None, cfg: Cfg = Cfg()):
    # everything drawn here is for the viewer; the agent never sees this image
    img = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    sx, sy = W / cfg.width, H / cfg.height
    if info is not None and info.get('seg') is not None:
        heat = cv2.resize(info['seg'].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        alpha = (np.clip(heat, 0.0, 1.0) * 0.5)[..., None]
        red = np.zeros_like(img)
        red[..., 2] = 255
        img = (img * (1 - alpha) + red * alpha).astype(np.uint8)
    for lb in boxes:
        p0 = (int(lb.x * sx), int(lb.y * sy))
        p1 = (int((lb.x + lb.width) * sx), int((lb.y + lb.height) * sy))
        cv2.rectangle(img, p0, p1, (0, 255, 0), 2)
    cx, cy = W // 2, H // 2
    cv2.line(img, (cx, cy - 8), (cx, cy + 8), (255, 255, 255), 1)
    cv2.line(img, (cx - 8, cy), (cx + 8, cy), (255, 255, 255), 1)
    probs = None if info is None else info.get('probs')
    for i, name in enumerate(ACTION_NAMES):
        y = 22 + 22 * i
        color = (0, 255, 255) if i == action else (200, 200, 200)
        if probs is not None:
            cv2.rectangle(img, (130, y - 12), (130 + int(150 * probs[i]), y + 2), color, -1)
            text = f'{name} {probs[i]:.2f}'
        else:
            text = name + (' <' if i == action else '')
        cv2.putText(img, text, (8, y), FONT, 0.5, color, 1, cv2.LINE_AA)
    if info is not None and info.get('vis') is not None:
        y = 22 + 22 * len(ACTION_NAMES) + 2  # below the action lines
        cv2.putText(img, f'enemy {info["vis"]:.2f}', (8, y), FONT, 0.5, (0, 200, 255), 1,
                    cv2.LINE_AA)
        if info['vis'] > 0.5:
            x = int((info['off'] + 1.0) / 2.0 * W)
            cv2.line(img, (x, 0), (x, H), (0, 200, 255), 2)
    if net_frame is not None:
        inset = cv2.cvtColor(np.ascontiguousarray(net_frame), cv2.COLOR_GRAY2BGR)
        h, w = inset.shape[:2]
        img[H - h - 8:H - 8, W - w - 8:W - 8] = inset
        cv2.rectangle(img, (W - w - 9, H - h - 9), (W - 8, H - 8), (255, 255, 255), 1)
        cv2.putText(img, 'network input', (W - w - 8, H - h - 14), FONT, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


class VideoSink:
    def __init__(self, path, fps, size):
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        assert self.writer.isOpened(), f'cannot open a video writer for {path}'

    def write(self, bgr):
        self.writer.write(bgr)

    def close(self):
        self.writer.release()


def video_frame_count(path):
    cap, n = cv2.VideoCapture(path), 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def play_episode(env, choose, seed, sink=None, window=False, speed=1.0, boxes_fn=None,
                 cfg: Cfg = Cfg(), every_tic=False):
    # every_tic: one frame per game tic (35 per second) instead of one per agent step
    if hasattr(choose, 'reset'):
        choose.reset()
    obs, total, n, done = env.reset(seed), 0.0, 0, False
    tic_s = 1.0 / (35 * speed)  # 35 game tics per second
    due = None

    def show(frame, tics):
        # write the frame, then hold it until the game clock says the next one is due
        nonlocal due
        if sink is not None:
            sink.write(frame)
        if not window:
            return
        cv2.imshow('DoomCNN', frame)
        now = time.perf_counter()
        if due is None or now - due > 0.1:  # first frame, or too late after a pause
            due = now
        due += tics * tic_s
        key = cv2.waitKey(max(1, int((due - now) * 1000))) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt
        if key == ord(' '):
            cv2.waitKey(0)
            due = None

    while not done:
        priv = env.privileged()
        action, info = choose(obs, priv)
        net = obs[-1]
        if every_tic:
            def tic(rgb, p):
                boxes = boxes_fn(p) if boxes_fn else ()
                show(draw_overlay(rgb, action, info, boxes, net, cfg), 1)
            obs, r, done = env.step(int(action), on_tic=tic)
        else:
            boxes = boxes_fn(priv) if boxes_fn else ()
            show(draw_overlay(env.screen(), action, info, boxes, net, cfg), cfg.frame_skip)
            obs, r, done = env.step(int(action))
        total += r
        n += 1
    return total, n
