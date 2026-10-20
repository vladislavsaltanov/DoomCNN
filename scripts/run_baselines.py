import os

import numpy as np

from doomagent.config import VAL_SEEDS
from doomagent.evaluate import evaluate_plan, summarize
from doomagent.reports import report_table_md
from doomagent.teacher import Teacher
from doomagent.variants import BASE

plan = [(BASE, VAL_SEEDS)]
rng = np.random.default_rng(0)
report = {'teacher': summarize(evaluate_plan(plan, Teacher())),
          'random': summarize(evaluate_plan(plan, lambda obs, priv: int(rng.integers(0, 3))))}
os.makedirs('reports', exist_ok=True)
with open('reports/s1_teacher.md', 'w') as fh:
    fh.write('# S1: teacher and random agent on val seeds (base graphics)\n\n'
             + report_table_md(report))
print(report_table_md(report))
