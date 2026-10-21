import json
import os

from doomagent.reports import report_table_md

F = 'bc_ppo_flip'  # the final model: Gate C met, best PPO finetune by val_score (S7)

with open('results/test_report.json') as fh:
    report = json.load(fh)
os.makedirs('reports', exist_ok=True)
parts = []
for split in ('id', 'ood'):
    rows = {k: v for k, v in report.items() if k.endswith('@' + split)}
    if rows:
        parts.append(f'## {split.upper()}\n\n' + report_table_md(rows))


def mean(name):
    return report[name]['reward']['mean']


# success criteria of the spec (plan Task 13 Step 6)
vs_t, vs_b = report[f'{F}@id']['vs_teacher'], report[f'{F}@ood']['vs_bc0']
checks = [
    (f'ID: upper bound of vs_teacher of {F}@id >= 0',
     f'{vs_t[0]:.2f} [{vs_t[1]:.2f}, {vs_t[2]:.2f}]', vs_t[2] >= 0),
    (f'ID: {F}@id >= scratch_rl@id', f"{mean(F + '@id'):.2f} vs {mean('scratch_rl@id'):.2f}",
     mean(F + '@id') >= mean('scratch_rl@id')),
    (f'ID: {F}@id >= scratch_total@id',
     f"{mean(F + '@id'):.2f} vs {mean('scratch_total@id'):.2f}",
     mean(F + '@id') >= mean('scratch_total@id')),
    (f'OOD: {F}@ood >= 0.8 * teacher@ood',
     f"{mean(F + '@ood'):.2f} vs 0.8 * {mean('teacher@ood'):.2f} = {0.8 * mean('teacher@ood'):.2f}",
     mean(F + '@ood') >= 0.8 * mean('teacher@ood')),
    (f'OOD: lower bound of vs_bc0 of {F}@ood > 0',
     f'{vs_b[0]:.2f} [{vs_b[1]:.2f}, {vs_b[2]:.2f}]', vs_b[1] > 0),
]
lines = ['## Success criteria', '', '| criterion | value | met |', '| --- | --- | --- |']
lines += [f'| {c} | {v} | {"yes" if ok else "NO"} |' for c, v, ok in checks]
lines += ['', '## Ablations (paired difference with bc0, mean [95% CI])', '',
          '| model | ID | OOD |', '| --- | --- | --- |']
for name in ('bc0_noaux', 'bc0_noseg', 'bc0_aug', 'r1', 'rB', 'bc_ppo', F):
    cells = []
    for split in ('id', 'ood'):
        d = report.get(f'{name}@{split}', {}).get('vs_bc0')
        cells.append('-' if d is None else '%.2f [%.2f, %.2f]' % tuple(d))
    lines.append(f'| {name} | {cells[0]} | {cells[1]} |')
lines += ['', f"bc0_aug@ood {mean('bc0_aug@ood'):.2f} vs r1@ood {mean('r1@ood'):.2f} "
              '(contribution of graphics variants beyond augmentations alone)']
parts.append('\n'.join(lines) + '\n')
with open('reports/s8_final.md', 'w', encoding='utf-8') as fh:
    fh.write('# S8: test-seed comparison\n\n' + '\n'.join(parts))
print('\n'.join(parts))
