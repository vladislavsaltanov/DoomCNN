"""S6 figure: val_score of every checkpoint of r1-r6, and the best checkpoint of each round on
the val seeds against the same checkpoint on fresh seeds (scripts/s6_compare.py).

    python scripts/s6_plot.py   ->   reports/s6_rounds.png
"""
import json
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from doomagent.reports import INK, MUTED, SURFACE  # noqa: E402

ROUNDS = [f'r{k}' for k in range(1, 7)]
# rounds are ordered: one blue ramp, light to dark (ordinal steps 250..700)
RAMP = ['#86b6ef', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
BLUE, ORANGE, GRID = '#2a78d6', '#eb6834', '#e4e3df'


def fresh_scores(path='reports/s6_compare.md'):
    # val_score column of the table written by scripts/s6_compare.py
    out = {}
    for line in open(path, encoding='utf-8'):
        m = re.match(r'\| (\w+) \| ([-0-9.]+) \|', line)
        if m:
            out[m[1]] = float(m[2])
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


if __name__ == '__main__':
    logs = {r: json.load(open(f'runs/{r}.pt.log.json'))['log'] for r in ROUNDS}
    fresh = fresh_scores()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE,
                               gridspec_kw=dict(width_ratios=[1.35, 1]))
    style(a)
    style(b)

    # left: every checkpoint of every round
    for r, c in zip(ROUNDS, RAMP):
        xs = [e['epoch'] for e in logs[r]]
        ys = [e['val_score'] for e in logs[r]]
        best = r == 'r4'  # the chosen round stands out; the others keep their order in the ramp
        a.plot(xs, ys, color=ORANGE if best else c, linewidth=3 if best else 1.6, marker='o',
               markersize=6 if best else 4,
               label=r + (' (лучший)' if best else ''), zorder=3 if best else 2)
    a.set_xlabel('эпоха обучения внутри раунда', color=INK, fontsize=10)
    a.set_ylabel('val_score', color=INK, fontsize=10)
    a.set_title('val_score каждого чекпойнта (30 val-seed: база и 8 вариантов)', color=INK,
                fontsize=11, loc='left')
    a.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=3, loc='lower right')

    # right: best checkpoint per round, selection seeds against fresh seeds
    k = range(1, 7)
    sel = [max(e['val_score'] for e in logs[r]) for r in ROUNDS]
    fr = [fresh[r] for r in ROUNDS]
    b.axhline(fresh['teacher'], color=MUTED, linewidth=1.2, linestyle='--', zorder=1)
    b.text(0.7, fresh['teacher'] + 0.15, f"учитель на свежих seed: {fresh['teacher']:.1f}",
           color=MUTED, fontsize=9, va='bottom')
    b.plot(k, sel, color=BLUE, linewidth=2, marker='o', markersize=7, label='на val-seed (выбор)')
    b.plot(k, fr, color=ORANGE, linewidth=2, marker='o', markersize=7,
           label='на свежих seed (проверка)')
    for x, ys, yf in zip(k, sel, fr):  # the higher value above its point, the lower below
        for y, up in ((ys, ys >= yf), (yf, yf > ys)):
            b.annotate(f'{y:.1f}', (x, y), textcoords='offset points', xytext=(0, 9 if up else -15),
                       color=INK, fontsize=8, ha='center')
    b.set_xticks(list(k), ROUNDS)
    b.set_xlim(0.6, 6.4)
    b.set_ylim(12.5, 21.2)
    b.set_ylabel('лучший val_score раунда', color=INK, fontsize=10)
    b.set_title('Лучший чекпойнт раунда: при выборе и на свежих seed', color=INK, fontsize=11,
                loc='left')
    b.legend(frameon=False, fontsize=9, labelcolor=INK, loc='lower right')
    fig.tight_layout()
    fig.savefig('reports/s6_rounds.png', dpi=150, facecolor=SURFACE)
    print('reports/s6_rounds.png')
