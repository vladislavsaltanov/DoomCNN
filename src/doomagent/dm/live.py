"""A live deathmatch for the DM dashboard: the teacher or a trained network plays in its own ViZDoom
on the CPU (training keeps the GPU). The teacher also plays in the shadow of every network step:
it sees the same obs and priv and gives its label, the network gets frames, vec and first only.
"""
import glob
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

from doomagent.dm.actions import HEADS, MOVE_NAMES, TURN_DEG
from doomagent.dm.config import DM, VARIANTS_DIR

JPEG = [cv2.IMWRITE_JPEG_QUALITY, 82]
WATCH_S = 3.0  # the network and memory views are drawn only while someone pulled them this recently
BRAIN_EVERY_S = 0.4  # trace() runs ten backward passes: at most this often
PLACE_EVERY_S = 6.0  # the live place maps are redrawn this often
AUTO = 'auto'        # the player 'the newest network': the freshest model file, reloaded at every match start


def newest_model():
    # the most recently written model: BC and DAgger models in runs/dm, a PPO run's latest.pt or best.pt, the E6 copies
    found = (glob.glob('runs/dm/*.pt') + glob.glob('runs/dm/*/latest.pt') + glob.glob('runs/dm/*/best.pt')
             + glob.glob('runs/dm/e6/*.pt'))
    return max(found, key=os.path.getmtime).replace(os.sep, '/') if found else 'teacher'


def describe(a):
    # 'forward · +10 · fire · next' for the page
    move, turn, fire, weapon = (int(x) for x in a)
    return ' · '.join([MOVE_NAMES[move], f'{TURN_DEG[turn]:+g}°'] + (['fire'] if fire else []) + (['next'] if weapon else []))


def _jpg(img):
    return cv2.imencode('.jpg', img, JPEG)[1].tobytes()


class DMLive:
    def __init__(self, cfg=DM, start=True):
        self.cfg, self.lock = cfg, threading.Lock()
        self.req = dict(model='teacher', variant='base', seed=35000, speed=1.0, running=False,
                        boxes=True, hud=True)
        self.screen = self.map = self.brain = self.memory = self.place = None
        # live place maps (dm.liveplace) of the model file playing: a fresher save of the same file keeps adding up
        self.places, self.places_of = None, None
        self.seen = dict(brain=0.0, memory=0.0)  # when a viewer last pulled each view
        self.state = dict(status='остановлено')
        self.history = deque(maxlen=30)
        self.series = deque(maxlen=1100)  # one point per step: a whole 2-minute match
        if start:
            threading.Thread(target=self._loop, daemon=True).start()

    def control(self, **kw):
        with self.lock:
            self.req.update(kw)
            self.req['changed'] = True

    def watched(self, which):
        return time.time() - self.seen.get(which, 0.0) < WATCH_S

    def _agent(self, path):
        if path == 'teacher':
            return None
        import torch
        torch.set_num_threads(2)
        from doomagent.dm.agent import DMAgent
        from doomagent.dm.model import load_model
        return DMAgent(load_model(path, self.cfg, map_location='cpu'))

    def _loop(self):
        from doomagent.dm.env import DeathmatchEnv
        from doomagent.dm.teacher import DMTeacher
        from doomagent.hud import Hud
        from doomagent.variants import get_variant
        env = agent = None
        env_name = loaded = None
        teacher, hud = DMTeacher(self.cfg), Hud()
        while True:
            with self.lock:
                req = dict(self.req)
                self.req['changed'] = False
            if not req['running']:
                self.state = dict(self.state, status='остановлено')
                time.sleep(0.2)
                continue
            try:
                if env is None or env_name != req['variant']:
                    if env is not None:
                        env.close()
                    self.state = dict(status='запуск ViZDoom...')
                    env = DeathmatchEnv(self.cfg, get_variant(req['variant'], VARIANTS_DIR))
                    env_name = req['variant']
                path = newest_model() if req['model'] == AUTO else req['model']
                key = (path, os.path.getmtime(path) if path != 'teacher' else 0)
                if key != loaded:
                    self.state = dict(status='загрузка модели...')
                    agent, loaded = self._agent(path), key
            except Exception as e:  # a bad model path or a broken variant
                self.state = dict(status=f'ошибка: {e}')
                with self.lock:
                    self.req['running'] = False
                continue
            self._match(env, agent, teacher, hud, dict(req, model=path), env_name)

    def _match(self, env, agent, teacher, hud, req, env_name):
        from doomagent.dm.brain import render, trace
        from doomagent.dm.memory_view import MemoryTape, events_from
        from doomagent.dm.minimap import MapView
        seed = int(req['seed'])
        obs, done, step = env.reset(seed), False, 0
        mapview = MapView(env.sectors)
        if self.places is None:
            from doomagent.dm.liveplace import PlaceMaps
            self.places = PlaceMaps(env.sectors)
        if self.places_of != req['model']:  # another model file: its place maps start over
            self.places.reset()
            self.places_of = req['model']
        last_place = time.time()
        teacher.reset()
        if agent is not None:
            agent.reset()
        memory = agent is not None and agent.model.memory
        tape = MemoryTape() if memory else None
        self.series.clear()
        with self.lock:
            self.memory = None
        agree = {h: 0 for h in HEADS + ('all',)}
        prev, last_brain = None, 0.0
        tic_s = 1.0 / (35 * max(0.1, float(req['speed'])))
        due = time.perf_counter()
        while not done:
            with self.lock:
                r2 = dict(self.req)
            if r2.get('changed') or not r2['running']:
                return
            priv = env.privileged()
            label = teacher(obs, priv)  # the shadow: same obs and priv as the player gets
            info = None
            if agent is None:
                action = label
            else:
                info = agent.explain(obs.frames, obs.vec, obs.first)
                action = info['action']
                if tape is not None:
                    tape.push(info['h'], info['z'], events_from(prev, priv, obs.first, step == 0))
                if self.watched('brain') and time.time() - last_brain > BRAIN_EVERY_S:
                    import torch
                    h = None if info['h_prev'] is None else torch.from_numpy(info['h_prev']).reshape(1, 1, 512)
                    b = _jpg(render(trace(agent.model, obs.frames, obs.vec, h), obs.frames, obs.vec))
                    last_brain = time.time()
                    with self.lock:
                        self.brain = b
                if tape is not None and self.watched('memory'):
                    m = _jpg(tape.render())
                    with self.lock:
                        self.memory = m
            prev = priv
            self.places.add(priv.px, priv.py, None if info is None else info['h'])
            if time.time() - last_place > PLACE_EVERY_S:
                pl = _jpg(self.places.render())
                last_place = time.time()
                with self.lock:
                    self.place = pl
            same = [int(a) == int(b) for a, b in zip(action, label)]
            for h, ok in zip(HEADS + ('all',), same + [all(same)]):
                agree[h] += ok

            def tic(rgb, p, action=action, info=info):
                nonlocal due
                from doomagent.dm.viewer import compose_view
                s = _jpg(compose_view(rgb, p, action, info, self.cfg, r2['boxes'],
                                      hud if r2['hud'] else None, None, teacher))
                mp = _jpg(mapview.draw(p, teacher))
                with self.lock:
                    self.screen, self.map = s, mp
                due += tic_s
                time.sleep(max(0.0, due - time.perf_counter()))
                if time.perf_counter() - due > 0.25:  # fell behind (a network view): no catch-up burst
                    due = time.perf_counter()

            obs, _, done = env.step(action, on_tic=tic)
            step += 1
            p = env.privileged()
            self.series.append(dict(step=step, agree=int(all(same)), agree_move=int(same[0]), agree_turn=int(same[1]),
                                    agree_fire=int(same[2]),
                                    p_teacher=None if info is None else float(info['probs'][1][int(label[1])]),
                                    frags=p.frags, health=max(p.health, 0.0)))
            self.state = dict(
                status='играет', model=req['model'], variant=env_name, seed=seed, step=step,
                frags=p.frags, deaths=p.deaths, kd=p.frags / max(p.deaths, 1), health=p.health, ammo=p.ammo,
                action=describe(action), teacher=describe(label), mode=teacher.mode, weapon=int(p.weapon),
                agree={h: n / step for h, n in agree.items()}, memory=bool(memory),
                probs=None if info is None else [[float(x) for x in pr] for pr in info['probs']])
        p = env.privileged()
        self.history.appendleft(dict(model=req['model'], variant=env_name, seed=seed, frags=p.frags,
                                     deaths=p.deaths, steps=step, agree=agree['all'] / max(step, 1)))
        with self.lock:
            if not self.req.get('changed'):
                self.req['seed'] = seed + 1  # the next match on the next seed
        time.sleep(1.0)
