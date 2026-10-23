import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from doomagent.dm.config import DM
from doomagent.dm.dashboard import dm_runs, dm_train, make_handler
from doomagent.dm.live import DMLive


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(DMLive(DM)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{srv.server_address[1]}'
    srv.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read().decode('utf-8')


def test_page_and_api(server):
    code, html = get(server + '/dm')
    assert code == 200 and 'deathmatch' in html
    assert 'teacher' in json.loads(get(server + '/api/options')[1])['models']
    assert json.loads(get(server + '/api/live')[1])['state']['status'] == 'остановлено'


def test_unknown_path_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server + '/live/secret.mjpg')
    assert e.value.code == 404


def test_runs_and_train_read_the_run_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'runs').mkdir()
    log = dict(config={}, log=[dict(epoch=5, val_score=3.5), dict(epoch=10, val_score=4.0)])
    (tmp_path / 'runs' / 'dm_m1_x.pt.log.json').write_text(json.dumps(log))
    (tmp_path / 'runs' / 'dm_m0_y.pt.progress.json').write_text(json.dumps(dict(epoch=1, points=[])))
    runs = {r['name']: r for r in dm_runs()}
    assert runs['dm_m1_x']['best'] == 4.0 and runs['dm_m0_y']['best'] is None
    d = dm_train('../runs/dm_m1_x')  # a path in the query string is cut to the run name
    assert d['log']['log'][-1]['epoch'] == 10 and len(d['actions']) == 10
