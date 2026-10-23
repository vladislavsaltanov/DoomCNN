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
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from doomagent.dm.config import ACTION_NAMES, DM, VARIANTS_DIR
from doomagent.dm.live import DMLive

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWS = ('screen', 'map', 'brain', 'memory')


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
    names = {_run_name(p) for p in glob.glob('runs/dm_*.pt.log.json') + glob.glob('runs/dm_*.pt.progress.json')}
    out = []
    for name in names:
        files = [f'runs/{name}.pt.{k}.json' for k in ('log', 'progress')]
        last = max((os.path.getmtime(f) for f in files if os.path.exists(f)), default=0)
        log = (_read_json(files[0]) or {}).get('log', [])
        best = max((r['val_score'] for r in log if 'val_score' in r), default=None)
        out.append(dict(name=name, updated=last, active=time.time() - last < 5 * 60, best=best,
                        epochs_done=log[-1]['epoch'] if log else 0))
    return sorted(out, key=lambda r: -r['updated'])


def dm_train(name):
    name = os.path.basename(name)  # no paths from the query string
    d = {k: _read_json(f'runs/{name}.pt.{k}.json') for k in ('progress', 'live', 'log')}
    teacher = (_read_json('reports/d1_teacher.json') or {}).get('report', {}).get('teacher', {}).get('frags')
    return dict(name=name, actions=list(ACTION_NAMES), teacher=teacher, **d)


def dm_options():
    models = sorted(glob.glob('runs/dm_*.pt'), key=os.path.getmtime, reverse=True)
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
                return self._send(dict(state=dm.state, req=req, history=list(dm.history),
                                       series=list(dm.series), actions=list(ACTION_NAMES)))
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
