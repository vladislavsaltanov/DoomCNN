"""The deathmatch dashboard, apart from the first section's panel: a live match (teacher or network),
the minimap, the network and its memory, and the training charts of the dm_* runs. Read-only towards
the project: it never writes to runs/ or data/.

    python -m doomagent.dm.dashboard            # then open http://127.0.0.1:8766
"""
import argparse
import ctypes
import glob
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from doomagent.dm.actions import HEADS, MOVE_NAMES, TURN_DEG
from doomagent.dm.config import DM, MODES, VARIANTS_DIR

# the values of every head, for the charts
VALUES = dict(move=list(MOVE_NAMES), turn=[f'{d:+g}°' for d in TURN_DEG], fire=['no', 'fire'], weapon=['keep', 'next'])
from doomagent.dm.live import DMLive

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWS = ('screen', 'map', 'brain', 'memory')
BOOT = str(time.time())  # a new value after a restart: the page then reconnects its picture streams


def _read_json(p):
    try:
        with open(p, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _run_name(p):
    return os.path.basename(p).split('.pt.')[0]


def dm_runs():
    # every dm_* training run: from its log (after the first evaluation) or its progress file
    names = {_run_name(p) for p in glob.glob('runs/dm/*.pt.log.json') + glob.glob('runs/dm/*.pt.progress.json')}
    out = []
    for name in names:
        files = [f'runs/dm/{name}.pt.{k}.json' for k in ('log', 'progress')]
        last = max((os.path.getmtime(f) for f in files if os.path.exists(f)), default=0)
        log = (_read_json(files[0]) or {}).get('log', [])
        best = max((r['val_score'] for r in log if 'val_score' in r), default=None)
        out.append(dict(name=name, updated=last, active=time.time() - last < 5 * 60, best=best,
                        epochs_done=log[-1]['epoch'] if log else 0))
    return sorted(out, key=lambda r: -r['updated'])


def dm_train(name):
    name = os.path.basename(name)  # no paths from the query string
    d = {k: _read_json(f'runs/dm/{name}.pt.{k}.json') for k in ('progress', 'live', 'log')}
    teacher = (_read_json('reports/dm/e1_teacher.json') or {}).get('report', {}).get('teacher', {}).get('frags')
    return dict(name=name, heads=list(HEADS), values=VALUES, teacher=teacher, **d)


_MATCH_CACHE = {}  # npz path -> (mtime, its numbers): finished matches are read once


def _match_numbers(path):
    import numpy as np
    m = os.path.getmtime(path)
    if path not in _MATCH_CACHE or _MATCH_CACHE[path][0] != m:
        z = np.load(path)
        a, c = z['actions'], z['counters']
        frames = path[:-4] + '.frames.npy'
        _MATCH_CACHE[path] = (m, dict(
            steps=len(a), frags=float(c[-1, 0]), deaths=float(c[-1, 1]), hits=float(c[-1, 2]),
            fire=int(a[:, 2].sum()), switch=int(a[:, 3].sum()), bot=int(z['bot_visible'].sum()),
            modes=np.bincount(z['modes'], minlength=len(MODES)).tolist(),
            weapons=np.bincount(z['weapon'].astype(int), minlength=8).tolist(), mtime=m,
            bytes=os.path.getsize(path) + (os.path.getsize(frames) if os.path.exists(frames) else 0)))
    return _MATCH_CACHE[path][1]


def dm_record():
    # the recording card: what is recorded, the finished matches and their numbers, the live workers
    from doomagent.dm.livetap import LIVE_DIR, STALE_S
    from doomagent.dm.weapons import WEAPONS
    plan = _read_json(os.path.join(LIVE_DIR, 'record.json'))
    now = time.time()
    workers = []
    for p in sorted(glob.glob(os.path.join(LIVE_DIR, '*.json'))):
        w = _read_json(p)
        if os.path.basename(p) != 'record.json' and w and now - w.get('time', 0) < STALE_S:
            workers.append(w)
    if not plan:
        return dict(plan=None, workers=workers)
    rows = []
    for p in glob.glob(os.path.join(plan['out'], 'ep_*.npz')):
        try:
            rows.append(_match_numbers(p))
        except (OSError, ValueError, KeyError):  # being written
            pass
    rows.sort(key=lambda r: r['mtime'])
    steps = sum(r['steps'] for r in rows)
    new = [r for r in rows if r['mtime'] >= plan['start']]
    rate = len(new) / max(now - plan['start'], 1.0)
    left = max(plan['total'] - len(rows), 0)
    weapons = [sum(r['weapons'][s] for r in rows) for s in range(8)]
    metrics = None
    if rows:
        metrics = dict(
            frags=sum(r['frags'] for r in rows) / len(rows), deaths=sum(r['deaths'] for r in rows) / len(rows),
            kd=sum(r['frags'] for r in rows) / max(sum(r['deaths'] for r in rows), 1.0),
            hits=sum(r['hits'] for r in rows) / len(rows), fire_share=sum(r['fire'] for r in rows) / steps,
            switch_share=sum(r['switch'] for r in rows) / steps, bot_share=sum(r['bot'] for r in rows) / steps,
            mode_share={m: sum(r['modes'][i] for r in rows) / steps for i, m in enumerate(MODES)},
            weapon_share={WEAPONS[s].name: weapons[s] / steps for s in WEAPONS if weapons[s]},
            frags_per_match=[r['frags'] for r in rows[-60:]])
    return dict(plan=plan, workers=workers, done=len(rows), steps=steps,
                gb=sum(r['bytes'] for r in rows) / 1e9, per_min=60 * rate,
                eta_s=left / rate if new and left else (0 if not left else None), metrics=metrics)


def dm_neurons(run):
    # neuron maps of this run's network kind (m0 or m1), newest first
    kind = os.path.basename(run).split('_')[0]
    files = sorted(glob.glob(f'artifacts/dm/*_select_{kind}.png'), key=os.path.getmtime, reverse=True)
    return dict(images=[dict(name=os.path.basename(f), mtime=os.path.getmtime(f)) for f in files])


def dm_options():
    models = sorted(glob.glob('runs/dm/*.pt'), key=os.path.getmtime, reverse=True)
    variants = ['base']
    try:
        from doomagent.variants import load_variants
        variants += [v.name for v in load_variants(root=VARIANTS_DIR)]
    except Exception:  # no deathmatch variants yet (D3)
        pass
    return dict(models=['teacher'] + [m.replace('\\', '/') for m in models], variants=variants)


def make_handler(dm, system=None):
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
                    dm.seen[which] = time.time()  # someone watches: keep drawing it
                    with dm.lock:
                        img = getattr(dm, which)
                    if img is not None and id(img) != seen:
                        seen = id(img)
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                         + str(len(img)).encode() + b'\r\n\r\n' + img + b'\r\n')
                    time.sleep(1 / 35)
            except (ConnectionError, OSError):
                return

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ('/', '/dm'):
                with open(os.path.join(HERE, 'dashboard_dm.html'), encoding='utf-8') as fh:
                    return self._send(fh.read(), 'text/html; charset=utf-8')
            if u.path.startswith('/live/') and u.path.endswith('.mjpg') and u.path[6:-5] in VIEWS:
                return self._mjpeg(u.path[6:-5])
            if u.path == '/api/live':
                with dm.lock:
                    req = {k: v for k, v in dm.req.items() if k != 'changed'}
                return self._send(dict(boot=BOOT, state=dm.state, req=req, history=list(dm.history),
                                       series=list(dm.series), heads=list(HEADS), values=VALUES))
            if u.path == '/api/control':
                kw = {k: q[k] for k in ('model', 'variant') if k in q}
                if 'seed' in q:
                    kw['seed'] = int(q['seed'])
                if 'speed' in q:
                    kw['speed'] = float(q['speed'])
                kw.update({k: q[k] == '1' for k in ('running', 'boxes', 'hud') if k in q})
                dm.control(**kw)
                return self._send({'ok': True})
            if u.path == '/api/options':
                return self._send(dm_options())
            if u.path == '/api/runs':
                return self._send(dm_runs())
            if u.path == '/api/train':
                return self._send(dm_train(q.get('run', '')))
            if u.path == '/api/record':
                return self._send(dm_record())
            if u.path == '/api/record/img':
                from doomagent.dm.livetap import LIVE_DIR
                w, k = q.get('w', ''), q.get('k', '')
                path = os.path.join(LIVE_DIR, f'{w}.{k}.jpg')
                if re.fullmatch(r'[\w-]+', w) and k in ('screen', 'map') and os.path.exists(path):
                    with open(path, 'rb') as fh:
                        return self._send(fh.read(), 'image/jpeg')
                return self._send({'error': 'not found'}, code=404)
            if u.path == '/api/neurons':
                return self._send(dm_neurons(q.get('run', '')))
            if u.path == '/artifact':
                name = q.get('name', '')
                path = os.path.join('artifacts', 'dm', name)
                if re.fullmatch(r'[\w.-]+\.png', name) and '..' not in name and os.path.exists(path):
                    with open(path, 'rb') as fh:
                        return self._send(fh.read(), 'image/png')
                return self._send({'error': 'not found'}, code=404)
            if u.path == '/api/system':
                return self._send(dict(now=system.now if system else {}))
            return self._send({'error': 'not found'}, code=404)

    return H


def main():
    from doomagent.dashboard import System
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8766)
    a = ap.parse_args()
    if os.name == 'nt':
        ctypes.windll.winmm.timeBeginPeriod(1)  # smooth 35 fps pacing
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), make_handler(DMLive(DM), System()))
    srv.daemon_threads = True
    print(f'deathmatch dashboard: http://127.0.0.1:{a.port}', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
