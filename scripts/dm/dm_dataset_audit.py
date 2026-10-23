"""Label sanity of recorded teacher matches: fire without a visible bot, steps the teacher stands stuck,
matches without frags, how often a label is played as is, turn labels with and without a bot in sight.

    python scripts/dm/dm_dataset_audit.py data/dm/r0 data/dm/r1
"""
import glob
import json
import os
import sys

import numpy as np

from doomagent.dm.actions import TURN_DEG
from doomagent.dm.config import MODES

STUCK_WINDOW = 16  # steps (1.8 s)
STUCK_MOVE = 32.0  # map units over the window

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    files = sorted(f for d in sys.argv[1:] for f in glob.glob(os.path.join(d, 'ep_*.npz')))
    n = fire = fire_nobot = stuck = played = 0
    frags, deaths, turn_bot, turn_nobot, fire_by_mode = [], [], [], [], np.zeros(len(MODES))
    steps_by_mode = np.zeros(len(MODES))
    for f in files:
        z = np.load(f)
        a, ex, bot, pose, modes = z['actions'], z['executed'], z['bot_visible'], z['pose'], z['modes']
        T = len(a)
        n += T
        fire += a[:, 2].sum()
        fire_nobot += (a[:, 2].astype(bool) & ~bot).sum()
        played += (a == ex).all(1).sum()
        step = np.hypot(np.diff(pose[:, 0]), np.diff(pose[:, 1]))
        step[step > 60] = 0.0  # a respawn
        win = np.convolve(step, np.ones(STUCK_WINDOW), 'valid')
        stuck += int((win < STUCK_MOVE).sum())
        frags.append(z['counters'][-1, 0])
        deaths.append(z['counters'][-1, 1])
        turn_bot += [abs(TURN_DEG[t]) for t in a[bot, 1]]
        turn_nobot += [abs(TURN_DEG[t]) for t in a[~bot, 1]]
        np.add.at(fire_by_mode, modes, a[:, 2])
        np.add.at(steps_by_mode, modes, 1)
    frags = np.array(frags)
    out = dict(matches=len(files), steps=n, fire_share=fire / n, fire_without_visible_bot=fire_nobot / max(fire, 1),
               label_played_as_is=played / n, stuck_window_share=stuck / n,
               frags=dict(mean=float(frags.mean()), min=float(frags.min()), p10=float(np.percentile(frags, 10)),
                          zero_matches=int((frags <= 0).sum())),
               deaths_mean=float(np.mean(deaths)),
               turn_abs_mean=dict(bot_visible=float(np.mean(turn_bot)), no_bot=float(np.mean(turn_nobot))),
               fire_share_by_mode={m: float(fire_by_mode[i] / max(steps_by_mode[i], 1)) for i, m in enumerate(MODES)})
    print(json.dumps(out, indent=1))
