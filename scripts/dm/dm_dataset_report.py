"""D3 dataset report: reports/dm/d3_dataset.{md,png,json} from the recorded teacher matches.

    python scripts/dm/dm_dataset_report.py data/dm/r0 data/dm/r1
"""
import argparse
import json

import numpy as np

from doomagent.dm.config import ACTION_NAMES, MODES
from doomagent.dm.reports import dataset_figure, dataset_stats

ap = argparse.ArgumentParser()
ap.add_argument('dirs', nargs='+')
ap.add_argument('--out', default='reports/dm/d3_dataset')
a = ap.parse_args()
per_dir = {d: dataset_stats([d]) for d in a.dirs}
total = dataset_stats(a.dirs)
dataset_figure(total, a.out + '.png')
with open(a.out + '.json', 'w') as fh:
    json.dump(dict(total=total, per_dir=per_dir), fh, indent=1)
lines = ['# D3: recorded teacher matches', '',
         '| set | matches | steps | GB | steps per match | lives per match | teacher frags per match | noise share | variants |',
         '|---|---|---|---|---|---|---|---|---|']
for d, s in list(per_dir.items()) + [('all', total)]:
    lines.append(f'| {d} | {s["matches"]} | {s["steps"]} | {s["gb"]:.1f} | {s["steps_per_match"]:.0f} | '
                 f'{np.mean(s["lives_per_match"]):.1f} | {np.mean(s["frags_per_match"]):.2f} | '
                 f'{s["noise_share"]:.3f} | {len(s["per_variant"])} |')
lines += ['', '| action | share of labels |', '|---|---|']
lines += [f'| {n} | {v:.1%} |' for n, v in zip(ACTION_NAMES, total['action_share'])]
lines += ['', '| teacher mode | share of steps |', '|---|---|']
lines += [f'| {n} | {v:.1%} |' for n, v in zip(MODES, total['mode_share'])]
lines += ['', f'![dataset]({a.out.split("/")[-1]}.png)', '']
with open(a.out + '.md', 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines))
print(f'written {a.out}.md, .png, .json: {total["matches"]} matches, {total["steps"]} steps, {total["gb"]:.1f} GB')
