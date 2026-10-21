"""S8 figure: test reward of every model with its 95% bootstrap interval, one split per figure.

    python scripts/s8_plot.py id    ->  reports/s8_id.png
    python scripts/s8_plot.py ood   ->  reports/s8_ood.png

Reads results/test_report.json, or the lines the running report prints (runs/s8_report.out).
"""
import json
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from doomagent.reports import INK, MUTED, SURFACE  # noqa: E402

BLUE, ORANGE, GRID, BAND = '#2a78d6', '#eb6834', '#e4e3df', '#dcdad3'
ORDER = [  # top to bottom: the method's steps, then the controls
    ('teacher', 'учитель (скрипт по данным игры)'),
    ('bc_ppo_flip', 'итоговая: r4 + PPO + штраф за кручение'),
    ('bc_ppo', 'r4 + PPO'),
    ('rB', 'r4: имитация + варианты + DAgger'),
    ('r1', 'r1: имитация + варианты графики'),
    ('bc0', 'bc0: имитация, только база'),
    ('bc0_aug', 'bc0_aug: только база + аугментации'),
    ('bc0_noaux', 'bc0_noaux: без вспомогательных голов'),
    ('bc0_noseg', 'bc0_noseg: без головы маски'),
    ('scratch_total', 'PPO с нуля, полный бюджет (446k)'),
    ('scratch_rl', 'PPO с нуля, бюджет RL (200k)'),
]
TITLE = {'id': 'Test, базовая графика: 100 эпизодов на модель',
         'ood': 'Test, незнакомая графика: 11 test-вариантов × 20 эпизодов'}


def load():
    if os.path.exists('results/test_report.json'):
        with open('results/test_report.json') as fh:
            return json.load(fh)
    out = {}
    for line in open('runs/s8_report.out', encoding='utf-8', errors='replace'):
        name, _, rest = line.partition(' ')
        if '@' in name and rest.startswith('{'):
            out[name] = json.loads(rest)
    return out


if __name__ == '__main__':
    split = sys.argv[1] if len(sys.argv) > 1 else 'id'
    rep = load()
    rows = [(k, label, rep[f'{k}@{split}']['reward']) for k, label in ORDER
            if f'{k}@{split}' in rep]
    fig, ax = plt.subplots(figsize=(10, 0.48 * len(rows) + 1.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    t = rep.get(f'teacher@{split}', {}).get('reward')
    if t:
        ax.axvspan(t['lo'], t['hi'], color=BAND, zorder=0)
        ax.text(t['hi'] + 0.15, len(rows) - 0.45, 'интервал учителя', color=MUTED, fontsize=8,
                va='top')
    for i, (k, label, r) in enumerate(rows):
        y = len(rows) - 1 - i
        color = ORANGE if k == 'bc_ppo_flip' else (MUTED if k == 'teacher' else BLUE)
        ax.plot([r['lo'], r['hi']], [y, y], color=color, linewidth=2.5, solid_capstyle='round')
        ax.plot(r['mean'], y, 'o', color=color, markersize=8, zorder=3)
        ax.text(r['hi'] + 0.25, y, f"{r['mean']:.2f}  [{r['lo']:.2f}, {r['hi']:.2f}]",
                color=INK, fontsize=9, va='center')
    ax.set_yticks(range(len(rows)), [label for _, label, _ in rows][::-1])
    ax.tick_params(axis='y', colors=INK, labelsize=9, length=0)
    ax.tick_params(axis='x', colors=MUTED, labelsize=9)
    lo = min(r['lo'] for _, _, r in rows)
    ax.set_xlim(min(lo - 1, -1.5), 29)
    ax.set_xlabel('средняя награда за эпизод (95% bootstrap-интервал); максимум ≈ 25', color=INK,
                  fontsize=9)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.set_title(TITLE.get(split, split), color=INK, fontsize=11, loc='left')
    fig.tight_layout()
    path = f'reports/s8_{split}.png'
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(path, len(rows), 'models')
