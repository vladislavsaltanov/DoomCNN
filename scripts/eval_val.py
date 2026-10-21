import json
import os
import sys

import numpy as np

from doomagent.agent import load_agent
from doomagent.evaluate import evaluate_plan, pixel_only, val_score
from doomagent.variants import val_plan

plan = val_plan()
os.makedirs('reports', exist_ok=True)
for path in sys.argv[1:]:
    res = evaluate_plan(plan, pixel_only(load_agent(path).act))
    score, per = val_score(res)
    ood = [r for k, r in per.items() if k != 'base']
    row = dict(model=path, val_score=score, base=per.get('base'),
               ood_mean=float(np.mean(ood)) if ood else None, per_variant=per)
    print(json.dumps(row))
    name = os.path.splitext(os.path.basename(os.path.dirname(path) if path.endswith('.zip')
                                             else path))[0]
    with open(f'reports/val_{name}.json', 'w') as fh:
        json.dump(row, fh, indent=1)
