"""What a recording worker plays right now, for the dashboard: every every_s seconds it writes its game
frame (label boxes, the teacher's action), the minimap and its state to runs/dm/live/<worker>.*.
Writes are atomic (temp file, then replace) and never stop the recording when a reader holds a file.
"""
import json
import os
import time

import cv2

from doomagent.dm.config import DM

LIVE_DIR = 'runs/dm/live'
STALE_S = 15.0  # a worker silent this long is no longer shown


def _put(path, data):
    tmp = path + '.tmp'
    try:
        with open(tmp, 'wb') as fh:
            fh.write(data)
        os.replace(tmp, path)
    except OSError:  # the dashboard is reading it (Windows): skip this frame
        pass


def _jpg(bgr_or_rgb, rgb=True):
    img = cv2.cvtColor(bgr_or_rgb, cv2.COLOR_RGB2BGR) if rgb else bgr_or_rgb
    return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def write_plan(name, out, total, root=LIVE_DIR):
    # what is being recorded: the dashboard counts the finished matches in out against total
    os.makedirs(root, exist_ok=True)
    _put(os.path.join(root, 'record.json'),
         json.dumps(dict(name=name, out=out.replace('\\', '/'), total=int(total), start=time.time())).encode())


class LiveTap:
    def __init__(self, root=LIVE_DIR, every_s=0.1, worker=None, cfg=DM):
        self.root, self.every_s, self.cfg = root, every_s, cfg
        self.worker = worker or f'w{os.getpid()}'
        self.last, self.mapview, self.sectors = -1e9, None, None
        os.makedirs(root, exist_ok=True)

    def __call__(self, env, priv, teacher, label, action, info):
        now = time.time()
        if now - self.last < self.every_s:
            return
        self.last = now
        from doomagent.dm.live import describe
        from doomagent.dm.viewer import compose_view
        base = os.path.join(self.root, self.worker)
        _put(base + '.screen.jpg', _jpg(compose_view(env.screen(), priv, action, None, self.cfg, True, None, None,
                                                     teacher), rgb=False))
        sectors = getattr(env, 'sectors', None)
        if sectors:
            if sectors is not self.sectors:
                from doomagent.dm.minimap import MapView
                self.mapview, self.sectors = MapView(sectors), sectors
            _put(base + '.map.jpg', _jpg(self.mapview.draw(priv, teacher), rgb=False))
        _put(base + '.json', json.dumps(dict(
            worker=self.worker, time=now, seed=int(info['seed']), variant=info['variant'], step=int(info['step']),
            frags=float(priv.frags), deaths=float(priv.deaths), health=float(priv.health), weapon=int(priv.weapon),
            mode=teacher.mode, why=getattr(teacher, 'why', ''), label=describe(label),
            action=describe(action))).encode())
