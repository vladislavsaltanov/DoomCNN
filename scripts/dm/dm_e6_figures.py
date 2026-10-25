"""E6 summary figures from the fresh-seed reports of E4-E6 (reports/dm/e6_ppo/):

- e6_models.png: frags, K/D and the stuck share of the teacher, BC, DAgger and the PPO candidates (95% CIs);
- e6_base_vs_variants.png: frags on the base graphics against the unfamiliar val variants;
- e6_ladder.png: the share of the teacher's frags after every step of the pipeline;
- e6_confirm.png: the three best PPO models on the 40 independent matches.

    python scripts/dm/dm_e6_figures.py
"""
import json

import numpy as np

from doomagent.dm.reports import ENTITY, MUTED, SURFACE, _style, models_figure, plt
from doomagent.reports import CATEGORICAL

R = 'reports/dm/'
REPORTS = [R + 'e4_bc/e4_fresh.json', R + 'e5_dagger/e5_fresh_sp.json', R + 'e6_ppo/e6_fresh_ppo.json']
OUT = R + 'e6_ppo/'
STEPS = [('dm3_m1_aux', 'BC\n(E4)'), ('dagger_sp_r2', 'DAgger\n(E5)'), ('ppo_kl_01509k', 'PPO\n1.5M'),
         ('ppo_kl_02265k', 'PPO\n2.27M'), ('ppo_kl_03000k', 'PPO\n3M')]


def load():
    models, per = {}, {}
    for p in REPORTS:
        with open(p) as fh:
            r = json.load(fh)
        for n, s in r['models'].items():
            models.setdefault(n, s)
        for n, ms in r['per_match'].items():
            per.setdefault(n, ms)
    return models, per


def base_vs_variants(per, out):
    names = ['teacher'] + [n for n, _ in STEPS]
    labels = ['teacher'] + [lab.replace('\n', ' ') for _, lab in STEPS]
    base = [np.mean([m['frags'] for m in per[n] if m['variant'] == 'base']) for n in names]
    var = [np.mean([m['frags'] for m in per[n] if m['variant'] != 'base']) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=SURFACE)
    for k, (vals, name, color) in enumerate(((base, 'base graphics (40 matches)', CATEGORICAL[1]),
                                             (var, 'unfamiliar graphics (20 matches)', CATEGORICAL[2]))):
        bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=color, label=name)
        ax.bar_label(bars, labels=['%.1f' % v for v in vals], color=MUTED, fontsize=8, padding=2)
    ax.set_xticks(x, labels, fontsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    _style(ax, 'Frags per match: base graphics against unfamiliar graphics (fresh seeds)')
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def ladder(models, out):
    t = models['teacher']['frags']['mean']
    share = [100 * models[n]['frags']['mean'] / t for n, _ in STEPS]
    lo = [100 * models[n]['frags']['lo'] / t for n, _ in STEPS]
    hi = [100 * models[n]['frags']['hi'] / t for n, _ in STEPS]
    x = np.arange(len(STEPS))
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=SURFACE)
    ax.plot(x, share, color=ENTITY['ppo'], linewidth=2.5, marker='o', markersize=9)
    ax.fill_between(x, lo, hi, color=ENTITY['ppo'], alpha=0.15, linewidth=0)
    for i, s in enumerate(share):
        ax.annotate('%.0f %%' % s, (i, s), xytext=(0, 10), textcoords='offset points', ha='center', color=MUTED,
                    fontsize=9)
    ax.axhline(100, color=ENTITY['teacher'], linewidth=1, linestyle='--')
    ax.axhline(50, color=MUTED, linewidth=0.8, linestyle=':')
    ax.annotate('teacher', (len(STEPS) - 1, 100), xytext=(6, 4), textcoords='offset points', color=MUTED, fontsize=8)
    ax.annotate('Gate B', (0, 50), xytext=(-4, 4), textcoords='offset points', color=MUTED, fontsize=8)
    ax.set_xticks(x, [lab for _, lab in STEPS], fontsize=9)
    ax.set_ylim(0, 115)
    _style(ax, "Share of the teacher's frags after each step (fresh seeds, 95% CI)")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def confirm(out):
    with open(R + 'e6_ppo/e6_confirm_ppo.json') as fh:
        r = json.load(fh)['models']
    names = ['ppo_kl_02013k', 'ppo_kl_02265k', 'ppo_kl_03000k']
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    for i, n in enumerate(names):
        f = r[n]['frags']
        ax.errorbar([i], [f['mean']], yerr=[[f['mean'] - f['lo']], [f['hi'] - f['mean']]], fmt='o',
                    color=CATEGORICAL[1 + i], capsize=4, markersize=8)
        ax.annotate('%.2f' % f['mean'], (i, f['mean']), xytext=(8, 0), textcoords='offset points', color=MUTED,
                    fontsize=8, va='center')
    ax.axhline(12.68, color=ENTITY['teacher'], linewidth=1, linestyle='--')
    ax.annotate('teacher on these matches (E3)', (0, 12.68), xytext=(0, 5), textcoords='offset points', color=MUTED,
                fontsize=8)
    ax.set_xticks(range(len(names)), ['2.01M', '2.27M (val)', '3M (final)'], fontsize=9)
    ax.set_xlim(-0.5, len(names) - 0.3)
    _style(ax, 'PPO models on 40 independent matches (base graphics)')
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    models, per = load()
    models_figure({n: models[n] for n, _ in STEPS} | {'ppo_kl_02013k': models['ppo_kl_02013k']},
                  OUT + 'e6_models.png', teacher=models['teacher'])
    base_vs_variants(per, OUT + 'e6_base_vs_variants.png')
    ladder(models, OUT + 'e6_ladder.png')
    confirm(OUT + 'e6_confirm.png')
    print('written', OUT + 'e6_{models,base_vs_variants,ladder,confirm}.png')
