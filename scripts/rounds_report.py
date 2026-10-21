import glob
import os
import re

from doomagent.reports import rounds_summary

paths = sorted(glob.glob('runs/r*.pt.log.json'),
               key=lambda p: int(re.search(r'r(\d+)\.pt', p).group(1)))
logs = {os.path.basename(p).split('.')[0]: p for p in paths}
rows, stop, best = rounds_summary(logs)
lines = ['# Rounds: best val_score per round', '', '| round | val_score | gain |',
         '| --- | --- | --- |']
lines += ['| %s | %.2f | %s |' % (r['name'], r['val_score'],
                                  '-' if r['gain'] is None else '%+.2f' % r['gain'])
          for r in rows]
lines += ['', f'- best: {best}', f'- stop rule met: {stop}']
os.makedirs('reports', exist_ok=True)
with open('reports/rounds.md', 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
