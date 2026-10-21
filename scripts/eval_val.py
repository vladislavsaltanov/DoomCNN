import argparse
import json
import os

import numpy as np
import torch

from doomagent.agent import StudentChooser, load_agent
from doomagent.evaluate import evaluate_plan, pixel_only, val_score
from doomagent.variants import val_plan

if __name__ == '__main__':  # worker processes re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='+')
    ap.add_argument('--workers', type=int, default=1,
                    help='processes for .pt students; results equal the serial run')
    a = ap.parse_args()
    plan = val_plan()
    os.makedirs('reports', exist_ok=True)
    for path in a.models:
        if a.workers > 1 and path.endswith('.pt'):
            chooser = StudentChooser(torch.load(path, map_location='cpu'))
        else:
            chooser = pixel_only(load_agent(path).act)
        res = evaluate_plan(plan, chooser, workers=a.workers if path.endswith('.pt') else 1)
        score, per = val_score(res)
        ood = [r for k, r in per.items() if k != 'base']
        row = dict(model=path, val_score=score, base=per.get('base'),
                   ood_mean=float(np.mean(ood)) if ood else None, per_variant=per)
        print(json.dumps(row))
        name = os.path.splitext(os.path.basename(os.path.dirname(path) if path.endswith('.zip')
                                                 else path))[0]
        with open(f'reports/val_{name}.json', 'w') as fh:
            json.dump(row, fh, indent=1)
