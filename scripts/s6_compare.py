"""S6: re-evaluate the best checkpoint of each round and the teacher on fresh seeds.

The best round is picked as a maximum over noisy checkpoints on the 30 val seeds, so its val
score is optimistic. Seeds 6000-6059 belong to no other set (train, DAgger, variants, val,
test, RL), so this is an unbiased second look on the same val variants.
"""
import json
import os
import sys

import numpy as np
import torch

from doomagent.agent import StudentChooser
from doomagent.evaluate import evaluate_plan, summarize, val_score
from doomagent.reports import report_table_md
from doomagent.teacher import Teacher
from doomagent.variants import BASE, load_variants

SEEDS = tuple(range(6000, 6060))
OOD_SEEDS = SEEDS[:20]

if __name__ == '__main__':
    models = sys.argv[1:] or [f'runs/r{k}.pt' for k in range(1, 7)]
    plan = [(BASE, SEEDS)] + [(v, OOD_SEEDS) for v in load_variants('val')]
    rows, report = {}, {}
    for name in ['teacher'] + models:
        chooser = Teacher() if name == 'teacher' else StudentChooser(
            torch.load(name, map_location='cpu'))
        res = evaluate_plan(plan, chooser, workers=4)
        score, per = val_score(res)
        key = os.path.splitext(os.path.basename(name))[0]
        ood = [v for k, v in per.items() if k != 'base']
        rows[key] = dict(val_score=score, base=per['base'], ood_mean=float(np.mean(ood)),
                         per_variant=per)
        report[key + '@base'] = summarize([r for r in res if r['variant'] == 'base'])
        report[key + '@ood'] = summarize([r for r in res if r['variant'] != 'base'])
        print(key, json.dumps({k: round(v, 2) for k, v in rows[key].items() if k != 'per_variant'}),
              flush=True)
    os.makedirs('reports', exist_ok=True)
    head = ['| model | val_score | base | ood_mean | ' + ' | '.join(rows['teacher']['per_variant']) + ' |']
    head.append('|' + ' --- |' * (4 + len(rows['teacher']['per_variant'])))
    for k, r in rows.items():
        head.append(f"| {k} | {r['val_score']:.2f} | {r['base']:.2f} | {r['ood_mean']:.2f} | "
                    + ' | '.join(f'{v:.1f}' for v in r['per_variant'].values()) + ' |')
    with open('reports/s6_compare.md', 'w', encoding='utf-8') as fh:
        fh.write('# S6: best checkpoint per round on fresh seeds\n\n'
                 f'base: seeds {SEEDS[0]}-{SEEDS[-1]}; each val variant: {OOD_SEEDS[0]}-{OOD_SEEDS[-1]}\n\n'
                 + '\n'.join(head) + '\n\n## Play metrics\n\n' + report_table_md(report))
    print(open('reports/s6_compare.md', encoding='utf-8').read())
