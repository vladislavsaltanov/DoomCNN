import os
from dataclasses import replace

from doomagent.config import VAL_SEEDS, Cfg
from doomagent.evaluate import evaluate_plan, summarize
from doomagent.reports import report_table_md
from doomagent.teacher import Teacher
from doomagent.variants import BASE

# a teacher that refuses to shoot targets narrower than w px at 160x120: if the reward barely
# drops, the network never has to aim at targets it can hardly see
report = {}
for w in (0.0, 3.0, 5.0, 8.0):
    cfg = replace(Cfg(), min_target_w=w)
    report[f'min_w={w:g}'] = summarize(evaluate_plan([(BASE, VAL_SEEDS)], Teacher(cfg), cfg))
os.makedirs('reports', exist_ok=True)
with open('reports/s1_resolution.md', 'w') as fh:
    fh.write('# S1: teacher reward vs minimal target width (px at 160x120)\n\n'
             + report_table_md(report))
print(report_table_md(report))
