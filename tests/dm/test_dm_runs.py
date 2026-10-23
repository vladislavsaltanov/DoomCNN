import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'dm'))

import dm_runs  # noqa: E402


def test_dm_runs_skips_finished(tmp_path):
    root = str(tmp_path)
    done = os.path.join(root, 'a.pt')
    open(done, 'wb').close()
    with open(done + '.log.json', 'w') as fh:
        json.dump(dict(log=[dict(epoch=12)]), fh)

    def fake(name, kind, dirs, flags, out, epochs):
        # writes a marker instead of training
        return [sys.executable, '-c', f'open({os.path.join(root, name + ".ran")!r}, "w").close()']

    jobs = [('a', 'm0', ['d'], []), ('b', 'm1', ['d'], [])]
    dm_runs.run_queue(jobs, root=root, epochs=12, parallel=1, make_cmd=fake, poll_s=0.05)
    assert not os.path.exists(os.path.join(root, 'a.ran'))
    assert os.path.exists(os.path.join(root, 'b.ran'))


def test_dm_runs_unfinished_log_runs_again(tmp_path):
    root = str(tmp_path)
    p = os.path.join(root, 'a.pt')
    open(p, 'wb').close()
    with open(p + '.log.json', 'w') as fh:
        json.dump(dict(log=[dict(epoch=6)]), fh)
    assert not dm_runs.finished(p, 12)
    with open(p + '.log.json', 'w') as fh:
        json.dump(dict(log=[dict(epoch=12)]), fh)
    assert dm_runs.finished(p, 12)
    os.remove(p)
    assert not dm_runs.finished(p, 12)  # a log without its model is not done
