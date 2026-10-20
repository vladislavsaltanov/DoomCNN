import os
import sys

from doomagent.reports import dataset_report, mask_preview

dirs = sys.argv[1:] or ['data/train']
tag = os.path.basename(os.path.normpath(dirs[-1]))
os.makedirs('reports', exist_ok=True)
os.makedirs('artifacts', exist_ok=True)
dataset_report(dirs, f'reports/dataset_{tag}.png', f'reports/dataset_{tag}.md')
mask_preview(dirs, f'artifacts/mask_{tag}.png')
