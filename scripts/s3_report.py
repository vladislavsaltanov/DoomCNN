import glob
import os

from doomagent.config import VAL_SEEDS
from doomagent.evaluate import evaluate_plan, summarize
from doomagent.reports import plot_bc_curves
from doomagent.teacher import Teacher
from doomagent.variants import BASE

logs = {os.path.basename(p)[:-len('.pt.log.json')]: p
        for p in sorted(glob.glob('runs/bc0*.pt.log.json'))}
t = summarize(evaluate_plan([(BASE, VAL_SEEDS)], Teacher()))['reward']
os.makedirs('reports', exist_ok=True)
plot_bc_curves(logs, 'reports/s3_curves.png', teacher=(t['lo'], t['hi']))
print(sorted(logs), 'teacher', t)
