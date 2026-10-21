"""Local demo dashboard: live student game, what the network sees, training curves, project
stage and machine load. Read-only towards the project: it never writes to runs/ or data/.

    python -m doomagent.dashboard            # then open http://127.0.0.1:8765
"""
import argparse
import ctypes
import glob
import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from ctypes import wintypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from doomagent.config import ACTION_NAMES, Cfg

HERE = os.path.dirname(os.path.abspath(__file__))
SLICES = [('S0', 'Видит ли сеть игру как задумано'), ('S1', 'Потолок учителя, разрешение'),
          ('S2', 'Датасет R0'), ('S3', 'Ученик повторяет учителя'),
          ('S4', 'Варианты графики'), ('S5', 'R1: обобщение на новую графику'),
          ('S6', 'Раунды DAgger'), ('S7', 'PPO выше имитации?'), ('S8', 'Итоговые критерии')]


# ---------------------------------------------------------------- machine load

class _FT(ctypes.Structure):
    _fields_ = [('lo', wintypes.DWORD), ('hi', wintypes.DWORD)]


class _MemStatus(ctypes.Structure):
    _fields_ = [('len', wintypes.DWORD), ('load', wintypes.DWORD)] + [
        (n, ctypes.c_ulonglong) for n in ('phys_total', 'phys_avail', 'commit_total',
                                          'commit_avail', 'virt_total', 'virt_avail', 'ext')]


def _ft(f):
    return (f.hi << 32) | f.lo


class System:
    # samples GPU, CPU, RAM and the commit charge every 2 s; keeps 10 minutes of history
    def __init__(self):
        self.now, self.hist, self._cpu = {}, deque(maxlen=300), None
        threading.Thread(target=self._loop, daemon=True).start()

    def _cpu_pct(self):
        if os.name != 'nt':
            return None
        idle, kern, user = _FT(), _FT(), _FT()
        ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern),
                                              ctypes.byref(user))
        cur = (_ft(idle), _ft(kern) + _ft(user))
        prev, self._cpu = self._cpu, cur
        if prev is None or cur[1] == prev[1]:
            return None
        return 100.0 * (1.0 - (cur[0] - prev[0]) / (cur[1] - prev[1]))

    @staticmethod
    def _mem():
        if os.name != 'nt':
            return {}
        m = _MemStatus()
        m.len = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        g = 2 ** 30
        return dict(ram_used=(m.phys_total - m.phys_avail) / g, ram_total=m.phys_total / g,
                    commit_used=(m.commit_total - m.commit_avail) / g,
                    commit_total=m.commit_total / g)

    @staticmethod
    def _gpu():
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,'
                 'temperature.gpu,power.draw', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5).stdout.strip().split(', ')
            return dict(gpu_name=out[0], gpu_util=float(out[1]), vram_used=float(out[2]) / 1024,
                        vram_total=float(out[3]) / 1024, gpu_temp=float(out[4]),
                        gpu_power=float(out[5]))
        except Exception:  # no NVIDIA driver, or nvidia-smi busy
            return {}

    def _loop(self):
        while True:
            d = dict(t=time.time(), cpu=self._cpu_pct(), **self._mem(), **self._gpu())
            if os.name == 'nt':
                for drive in ('C', 'E'):
                    free = ctypes.c_ulonglong()
                    if ctypes.windll.kernel32.GetDiskFreeSpaceExW(drive + ':\\', ctypes.byref(free),
                                                                  None, None):
                        d['disk_' + drive] = free.value / 2 ** 30
            self.now = d
            self.hist.append(d)
            time.sleep(2)


# ---------------------------------------------------------------- project state

def _read_json(p):
    try:
        with open(p, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _tail(p, n=12):
    try:
        with open(p, encoding='utf-8', errors='replace') as fh:
            return fh.read().splitlines()[-n:]
    except OSError:
        return []


def runs_state():
    out = []
    for p in sorted(glob.glob('runs/*.pt.log.json')):
        d = _read_json(p)
        if not d:
            continue
        name = os.path.basename(p)[:-len('.pt.log.json')]
        outf = f'runs/{name}.out'
        last = max(os.path.getmtime(p), os.path.getmtime(outf) if os.path.exists(outf) else 0)
        out.append(dict(name=name, config=d.get('config', {}), log=d.get('log', []),
                        updated=last, active=time.time() - last < 15 * 60
                        and len(d.get('log', [])) and d['log'][-1]['epoch'] < d['config'].get('epochs', 0)))
    return out


def train_progress():
    # the newest batch-level progress file written by train_bc, if it is fresh
    files = sorted(glob.glob('runs/*.pt.progress.json'), key=os.path.getmtime)
    if not files or time.time() - os.path.getmtime(files[-1]) > 10 * 60:
        return None
    d = _read_json(files[-1])
    if d:
        d['name'] = os.path.basename(files[-1])[:-len('.pt.progress.json')]
    return d


def project_state():
    log = ''
    try:
        with open('research_log.md', encoding='utf-8') as fh:
            log = fh.read()
    except OSError:
        pass
    heads = re.findall(r'^## (\S+), (S\d)[^\n]*', log, re.M)
    seen = {s for _, s in heads}
    entries = [dict(date=d, title=t) for d, t in re.findall(r'^## (\S+), ([^\n]+)', log, re.M)]
    datasets = []
    for p in sorted(glob.glob('data/*/meta.json')):
        m = _read_json(p) or {}
        datasets.append(dict(name=os.path.basename(os.path.dirname(p)),
                             episodes=m.get('episodes'), steps=m.get('steps'),
                             variants=len(m.get('variants', {}))))
    recording = []
    for d in sorted(glob.glob('data/*/')):
        if not os.path.exists(os.path.join(d, 'meta.json')):
            recording.append(dict(name=os.path.basename(d.rstrip('/\\')),
                                  episodes=len(glob.glob(os.path.join(d, 'ep_*.npz')))))
    rounds = _tail('reports/rounds.md', 30)
    s6 = _tail('runs/s6.log', 15)
    # the first slice without a log entry; S6 is in progress once its runner has written a log
    current = 'S6' if s6 and 'S6' not in seen else next((s for s, _ in SLICES if s not in seen),
                                                          None)
    return dict(slices=[dict(id=s, q=q, done=s in seen and s != current, current=s == current)
                        for s, q in SLICES],
                entries=entries[-8:], datasets=datasets, recording=recording,
                rounds=rounds, s6_log=s6)


GOAL = ('Агент играет в defend_the_center только по пикселям на уровне учителя '
        '(≈22.7 из 25 на базовой графике) и не разваливается на незнакомой графике.')
SLICE_GOALS = {
    'S6': 'Ученик сам играет на вариантах графики, учитель размечает его ошибки, ученик '
          'дообучается. Раунды идут, пока лучший val_score растёт больше чем на 0.5 '
          '(стоп: два раунда без роста или 5 раундов). Лучший раунд пойдёт в PPO (S7).',
    'S7': 'Дообучение с подкреплением (PPO): берём, только если val_score выше имитации.',
    'S8': 'Финальная проверка на test-seed и test-вариантах, отчёт по критериям спека.'}
ROUND_EPOCHS, ROUND_EPISODES, LAST_ROUND = 8, 100, 6


class Activity:
    # what the S6 runner is doing now and how long it will take, from runs/s6.log and the
    # training log of the running round
    @staticmethod
    def _events():
        out = []
        for line in _tail('runs/s6.log', 400):
            m = re.match(r'(\d\d):(\d\d):(\d\d) (.*)', line)
            if not m:
                continue
            h, mi, s, text = int(m[1]), int(m[2]), int(m[3]), m[4]
            t = time.mktime(time.localtime()[:3] + (h, mi, s, 0, 0, -1))
            if t > time.time() + 60:
                t -= 86400  # yesterday
            if text.startswith('restart'):
                out = []
            out.append((t, text))
        return out

    def __call__(self):
        now = time.time()
        ev = self._events()
        d = dict(goal=GOAL, slice_goal=SLICE_GOALS.get('S6'), phase='idle', now=now)
        if not ev:
            d['text'] = 'Сейчас ничего не запущено.'
            return d
        t0 = ev[0][0]
        d['s6_elapsed'] = now - t0
        last_t, last = ev[-1]
        if 'finished' in last or 'stop rule' in last:
            d.update(phase='done', text='S6 завершён: ' + last)
            return d
        m = re.match(r'round (\d+): (record|train)', last)
        if not m:
            d['text'] = last
            return d
        k, phase = int(m[1]), m[2]
        d.update(round=k, phase=phase, phase_elapsed=now - last_t,
                 rounds_left_max=LAST_ROUND - k)
        # finished rounds since the (re)start give the round time
        starts = [(t, int(re.match(r'round (\d+)', x)[1])) for t, x in ev
                  if re.match(r'round \d+: record', x)]
        done_rounds = [(b[0] - a[0]) for a, b in zip(starts, starts[1:])]
        round_s = sum(done_rounds) / len(done_rounds) if done_rounds else None
        if phase == 'record':
            n = len(glob.glob(f'data/dagger_r{k}/ep_*.npz'))
            rate = (now - last_t) / n if n else None
            left = rate * (ROUND_EPISODES - n) if rate else None
            d.update(progress=n / ROUND_EPISODES, phase_left=left,
                     text=f'Раунд {k}: ученик r{k - 1} играет 100 эпизодов на случайных '
                          f'вариантах графики, учитель размечает каждый шаг. Записано {n} из 100.')
        else:
            path = f'runs/r{k}.pt.log.json'
            log = (_read_json(path) or {}).get('log', [])
            ep = log[-1]['epoch'] if log else 0
            t_eval = os.path.getmtime(path) if log else None  # the log is rewritten after each eval
            per_epoch = (t_eval - last_t) / ep if ep else None
            left = per_epoch * (ROUND_EPOCHS - ep) if per_epoch else None
            # between evaluations, estimate the running epoch from the measured pace
            est = min(ROUND_EPOCHS, ep + (now - t_eval) / per_epoch) if ep else None
            prog = _read_json(f'runs/r{k}.pt.progress.json')
            if prog and os.path.getmtime(f'runs/r{k}.pt.progress.json') >= last_t:
                est = prog['epoch'] - 1 + prog['batch'] / max(prog['n_batches'], 1)
                frac = est / ROUND_EPOCHS
                if frac > 0.03:  # elapsed time already includes the evaluations so far
                    left = (now - last_t) * (1 - frac) / frac
            dirs = last.split(' on ', 1)[1].split() if ' on ' in last else []
            steps = sum((_read_json(os.path.join(p, 'meta.json')) or {}).get('steps', 0)
                        for p in dirs)
            d.update(progress=(est or ep) / ROUND_EPOCHS, phase_left=left, epoch=ep,
                     text=f'Раунд {k}: ученик r{k} дообучается от r{k - 1} на всех данных '
                          + (f'({steps:,} примеров) '.replace(',', ' ') if steps else '')
                          + f'{ROUND_EPOCHS} эпох; каждые 2 эпохи он играет 110 проверочных '
                          f'эпизодов (база и 8 val-вариантов). '
                          + (f'Сейчас эпоха {min(int(est) + 1, ROUND_EPOCHS)} из {ROUND_EPOCHS}'
                             f' ({est / ROUND_EPOCHS:.0%} раунда), проверено эпох: {ep}.'
                             if est is not None else f'Готово эпох: {ep} из {ROUND_EPOCHS}.'))
            if left is not None and round_s is None:
                round_s = (now - last_t) + left + 120  # this round, plus recording the next one
        if round_s:
            this_left = d.get('phase_left') or 0
            if phase == 'record':
                this_left += round_s * 0.75  # training is most of a round
            d['s6_left_max'] = this_left + round_s * (LAST_ROUND - k)
            d['round_s'] = round_s
        return d


def variants_list():
    from doomagent.variants import load_variants
    vs = [dict(name='base', split='base', kind='base')]
    vs += [dict(name=v.name, split=v.split, kind=v.kind) for v in load_variants()]
    return vs


# ---------------------------------------------------------------- live game

class Live:
    # plays one model in its own ViZDoom instance on the CPU, so training keeps the GPU
    def __init__(self, cfg: Cfg):
        self.cfg, self.lock = cfg, threading.Lock()
        self.req = dict(model='runs/r1.pt', variant='base', seed=5000, speed=1.0,
                        boxes=True, running=False)
        self.screen = self.net = None
        self.version = 0
        self.state = dict(status='остановлено')
        self.history = deque(maxlen=30)
        self.series = deque(maxlen=400)  # per step: network vs teacher and vs the game's truth
        threading.Thread(target=self._loop, daemon=True).start()

    def control(self, **kw):
        with self.lock:
            self.req.update(kw)
            self.req['changed'] = True

    def _chooser(self, model):
        if model == 'teacher':
            return None
        import torch
        torch.set_num_threads(2)
        from doomagent.model import StudentCNN
        m = StudentCNN(self.cfg)
        m.load_state_dict(torch.load(model, map_location='cpu'))
        m.eval()
        return m

    def _publish(self, rgb, info, boxes, obs):
        c = self.cfg
        img = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        H, W = img.shape[:2]
        if info and info.get('seg') is not None:
            heat = cv2.resize(info['seg'].astype(np.float32), (W, H))
            a = (np.clip(heat, 0, 1) * 0.55)[..., None]
            img = (img * (1 - a) + np.array([0, 0, 255]) * a).astype(np.uint8)
        sx, sy = W / c.width, H / c.height
        for lb in boxes:
            cv2.rectangle(img, (int(lb.x * sx), int(lb.y * sy)),
                          (int((lb.x + lb.width) * sx), int((lb.y + lb.height) * sy)),
                          (0, 255, 0), 1)
        if info and info.get('vis') is not None and info['vis'] > 0.5:
            x = int((info['off'] + 1) / 2 * W)
            cv2.line(img, (x, 0), (x, H), (0, 200, 255), 1)
        cx, cy = W // 2, H // 2
        cv2.line(img, (cx, cy - 8), (cx, cy + 8), (255, 255, 255), 1)
        cv2.line(img, (cx - 8, cy), (cx + 8, cy), (255, 255, 255), 1)
        frames = [cv2.resize(f, (c.width * 2, c.height * 2), interpolation=cv2.INTER_NEAREST)
                  for f in obs]
        tiles = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in frames]
        if info and info.get('seg') is not None:
            seg = cv2.resize((np.clip(info['seg'], 0, 1) * 255).astype(np.uint8),
                             (c.width * 2, c.height * 2), interpolation=cv2.INTER_NEAREST)
            tiles.append(cv2.applyColorMap(seg, cv2.COLORMAP_INFERNO))
        sep = np.full((c.height * 2, 6, 3), 30, np.uint8)
        row = tiles[0]
        for t in tiles[1:]:
            row = np.concatenate([row, sep, t], axis=1)
        s = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()
        n = cv2.imencode('.jpg', row, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        with self.lock:
            self.screen, self.net, self.version = s, n, self.version + 1

    def _loop(self):
        from doomagent.env import DoomEnv
        from doomagent.teacher import Teacher, enemies_from_labels, is_enemy_name, nearest_aux
        from doomagent.variants import get_variant
        env = model = None
        env_name = model_name = None
        teacher = Teacher(self.cfg)
        while True:
            with self.lock:
                req = dict(self.req)
                self.req['changed'] = False
            if not req['running']:
                self.state['status'] = 'остановлено'
                time.sleep(0.2)
                continue
            try:
                if env is None or env_name != req['variant']:
                    if env is not None:
                        env.close()
                    self.state['status'] = 'запуск ViZDoom...'
                    env, env_name = DoomEnv(self.cfg, get_variant(req['variant'])), req['variant']
                if model_name != req['model']:
                    self.state['status'] = 'загрузка модели...'
                    model, model_name = self._chooser(req['model']), req['model']
            except Exception as e:  # a bad model path or a broken variant
                self.state = dict(status=f'ошибка: {e}')
                with self.lock:
                    self.req['running'] = False
                continue
            seed = int(req['seed'])
            obs, total, step, done = env.reset(seed), 0.0, 0, False
            teacher.reset()
            agree = steps_cmp = 0
            self.series.clear()
            kills = 0
            tic_s = 1.0 / (35 * max(0.1, float(req['speed'])))
            due = time.perf_counter()
            while not done:
                with self.lock:
                    r2 = dict(self.req)
                if r2.get('changed') or not r2['running']:
                    break
                priv = env.privileged()
                label = teacher(obs, priv)
                if model is None:
                    action, info = label, None
                else:
                    info = model.explain(obs)
                    action = info['action']
                agree += action == label
                steps_cmp += 1
                boxes = ([lb for lb in priv.labels if is_enemy_name(lb.object_name)]
                         if r2['boxes'] else [])

                def tic(rgb, p, info=info, boxes=boxes, obs=obs):
                    nonlocal due
                    self._publish(rgb, info, boxes, obs)
                    due += tic_s
                    time.sleep(max(0.0, due - time.perf_counter()))
                    if time.perf_counter() - due > 0.25:
                        due = time.perf_counter()

                vis_t, off_t = nearest_aux(enemies_from_labels(priv.labels, priv.px, priv.py,
                                                               self.cfg.width))
                self.series.append(dict(
                    step=step, agree=int(action == label),
                    p_teacher=None if info is None else float(info['probs'][int(label)]),
                    vis_p=None if info is None else info['vis'], vis_t=vis_t,
                    off_p=None if info is None or info['vis'] < 0.5 else info['off'],
                    off_t=off_t if vis_t else None))
                obs, r, done = env.step(int(action), on_tic=tic)
                total += r
                kills += max(r, 0)
                step += 1
                self.state = dict(
                    status='играет', model=model_name, variant=env_name, seed=seed, step=step,
                    reward=total, kills=kills, health=priv.health, ammo=priv.ammo,
                    action=ACTION_NAMES[int(action)], teacher=ACTION_NAMES[int(label)],
                    agree=agree / max(steps_cmp, 1),
                    probs=None if info is None else [float(p) for p in info['probs']],
                    vis=None if info is None else info['vis'],
                    off=None if info is None else info['off'],
                    enemies=len(boxes) if r2['boxes'] else None)
            if done:
                self.history.appendleft(dict(model=model_name, variant=env_name, seed=seed,
                                             reward=total, steps=step,
                                             agree=agree / max(steps_cmp, 1)))
                with self.lock:
                    if not self.req.get('changed'):
                        self.req['seed'] = seed + 1  # next episode on the next seed
                time.sleep(1.0)


# ---------------------------------------------------------------- http

def make_handler(system, live, activity):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype='application/json', code=200):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _mjpeg(self, which):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            seen = -1
            try:
                while True:
                    with live.lock:
                        v, img = live.version, getattr(live, which)
                    if img is not None and v != seen:
                        seen = v
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                         + str(len(img)).encode() + b'\r\n\r\n' + img + b'\r\n')
                    time.sleep(1 / 35)
            except (ConnectionError, OSError):
                return

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == '/':
                with open(os.path.join(HERE, 'dashboard.html'), encoding='utf-8') as fh:
                    return self._send(fh.read(), 'text/html; charset=utf-8')
            if u.path == '/live/screen.mjpg':
                return self._mjpeg('screen')
            if u.path == '/live/net.mjpg':
                return self._mjpeg('net')
            if u.path == '/api/system':
                return self._send(dict(now=system.now, hist=list(system.hist)))
            if u.path == '/api/trainlive':
                return self._send(train_progress() or {})
            if u.path == '/api/activity':
                return self._send(activity())
            if u.path == '/api/project':
                return self._send(project_state())
            if u.path == '/api/runs':
                return self._send(runs_state())
            if u.path == '/api/live':
                with live.lock:
                    req = {k: v for k, v in live.req.items() if k != 'changed'}
                return self._send(dict(state=live.state, req=req, history=list(live.history),
                                       series=list(live.series)))
            if u.path == '/api/options':
                models = ['teacher'] + sorted(glob.glob('runs/*.pt'), key=os.path.getmtime,
                                              reverse=True)
                return self._send(dict(models=[m.replace('\\', '/') for m in models],
                                       variants=variants_list()))
            if u.path == '/api/control':
                kw = {}
                for k in ('model', 'variant'):
                    if k in q:
                        kw[k] = q[k]
                if 'seed' in q:
                    kw['seed'] = int(q['seed'])
                if 'speed' in q:
                    kw['speed'] = float(q['speed'])
                for k in ('running', 'boxes'):
                    if k in q:
                        kw[k] = q[k] == '1'
                live.control(**kw)
                return self._send({'ok': True})
            if u.path.startswith('/file/'):
                p = os.path.normpath(u.path[len('/file/'):])
                if p.startswith(('artifacts', 'reports')) and p.endswith('.png') \
                        and os.path.exists(p):
                    with open(p, 'rb') as fh:
                        return self._send(fh.read(), 'image/png')
            return self._send({'error': 'not found'}, code=404)

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8765)
    a = ap.parse_args()
    if os.name == 'nt':
        ctypes.windll.winmm.timeBeginPeriod(1)  # smooth 35 fps pacing
    system, live = System(), Live(Cfg())
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), make_handler(system, live, Activity()))
    srv.daemon_threads = True
    print(f'dashboard: http://127.0.0.1:{a.port}', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
