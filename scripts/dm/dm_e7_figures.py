"""E7 figures beyond dm_e7_report.py (reports/dm/e7_test/):

- e7_variants.png: frags on every test variant (the teacher, the final model, E6d, E6b, the base-only control);
- e7_memory.png: the memory question — BC M0 against M1, the final model with and without its memory, per variant;
- e7_models.png: frags, K/D and the stuck share of every player (95% CIs);
- e7_weapons.png: time with each weapon in hand, the teacher against the students;
- e7_play.png: how the players play — hits per shot, items picked up, path, share of steps with a bot in sight, first of 8.

    python scripts/dm/dm_e7_figures.py
"""
import json

import numpy as np

from doomagent.dm.reports import ENTITY, MUTED, SURFACE, _style, models_figure, plt
from doomagent.reports import CATEGORICAL

D = 'reports/dm/e7_test/'
FINAL = 'ppo_e6c_01001k'
LABEL = {'teacher': 'teacher', 'random': 'random', 'e7_m1_r0': 'BC base only', 'dm3_m0_aux': 'BC M0',
         'dm3_m1_aux': 'BC M1', 'dagger_sp_r2': 'DAgger', 'ppo_kl_03000k': 'PPO E6', 'ppo_explore_01001k': 'PPO E6b',
         FINAL: 'PPO E6c (final)', 'ppo_e6d_00503k': 'PPO E6d', FINAL + '_nomem': 'E6c, memory zeroed'}


def load():
    models, per = {}, {}
    for f in ('e7_test.json', 'e7_test_control.json'):
        with open(D + f) as fh:
            r = json.load(fh)
        models.update(r['models'])
        for n, ms in r['per_match'].items():
            per[n] = ms
    return models, per


def by_variant(per, name):
    out = {}
    for m in per[name]:
        out.setdefault(m['variant'], []).append(m['frags'])
    return {v: float(np.mean(x)) for v, x in out.items()}


def grouped(ax, groups, series, title, colors):
    x = np.arange(len(groups))
    w = 0.8 / len(series)
    for k, (name, vals) in enumerate(series):
        b = ax.bar(x + (k - (len(series) - 1) / 2) * w, vals, w, color=colors[k], label=name)
        ax.bar_label(b, labels=['%.1f' % v for v in vals], color=MUTED, fontsize=6, padding=1)
    ax.set_xticks(x, groups, fontsize=8, rotation=15)
    ax.legend(frameon=False, fontsize=7, labelcolor=MUTED, ncol=min(len(series), 5))
    _style(ax, title)


def variants_figure(per, out):
    vs = ['base', 'tex_test_00', 'tex_test_01', 'spr_test_sswv', 'spr_test_vile', 'mix_test_00_vile', 'mix_test_01_sswv']
    names = ['teacher', FINAL, 'ppo_e6d_00503k', 'ppo_explore_01001k', 'e7_m1_r0']
    colors = [ENTITY['teacher'], CATEGORICAL[1], CATEGORICAL[2], CATEGORICAL[3], ENTITY['bots']]
    fig, ax = plt.subplots(figsize=(15, 4.8), facecolor=SURFACE)
    grouped(ax, vs, [(LABEL[n], [by_variant(per, n).get(v, np.nan) for v in vs]) for n in names],
            'Frags per match on every test variant: base 40 matches, each variant 8 (tex = textures, spr = bot sprites, '
            'mix = both)', colors)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def memory_figure(per, models, out):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6), facecolor=SURFACE, gridspec_kw=dict(width_ratios=(1, 2.2)))
    names = ['dm3_m0_aux', 'dm3_m1_aux']
    m = [models[n]['frags'] for n in names]
    ax[0].bar([LABEL[n] for n in names], [s['mean'] for s in m], color=[CATEGORICAL[4], CATEGORICAL[1]],
              yerr=[[s['mean'] - s['lo'] for s in m], [s['hi'] - s['mean'] for s in m]], capsize=4, ecolor=MUTED)
    _style(ax[0], 'BC, equal data and budget: no memory (M0) vs GRU (M1)')
    vs = ['base', 'tex_test_00', 'tex_test_01', 'spr_test_sswv', 'spr_test_vile', 'mix_test_00_vile', 'mix_test_01_sswv']
    grouped(ax[1], vs, [(LABEL[n], [by_variant(per, n).get(v, np.nan) for v in vs]) for n in (FINAL, FINAL + '_nomem')],
            'The final model with its GRU memory and with the memory zeroed at every step', [CATEGORICAL[1], MUTED])
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def weapons_figure(models, out):
    from doomagent.dm.weapons import WEAPONS
    names = ['teacher', 'dagger_sp_r2', 'ppo_kl_03000k', FINAL]
    slots = [s for s in range(1, 8)]
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=SURFACE)
    grouped(ax, [WEAPONS[s].name if s in WEAPONS else str(s) for s in slots],
            [(LABEL[n], [100 * models[n]['weapon_share'][str(s)] for s in slots]) for n in names],
            'Share of the match with each weapon in hand, % (all 88 test matches)',
            [ENTITY['teacher'], CATEGORICAL[3], CATEGORICAL[2], CATEGORICAL[1]])
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def play_figure(models, out):
    names = ['teacher', 'dm3_m1_aux', 'dagger_sp_r2', 'ppo_kl_03000k', FINAL, 'ppo_e6d_00503k']
    keys = [('hits_per_shot', 'hits per shot', 1), ('items', 'items picked up per match', 1),
            ('path', 'path per match (map units)', 1), ('enemy_share', 'share of steps with a bot in sight', 1),
            ('deaths', 'deaths per match', 1)]
    fig, ax = plt.subplots(1, len(keys) + 1, figsize=(20, 4.2), facecolor=SURFACE)
    colors = [ENTITY['teacher'], CATEGORICAL[3], CATEGORICAL[4], CATEGORICAL[2], CATEGORICAL[1], CATEGORICAL[5]]
    for a, (k, title, _) in zip(ax, keys):
        vals = [models[n][k]['mean'] for n in names]
        b = a.bar(range(len(names)), vals, color=colors)
        a.bar_label(b, labels=['%.2f' % v if v < 10 else '%.0f' % v for v in vals], color=MUTED, fontsize=6, padding=1)
        a.set_xticks(range(len(names)), [LABEL[n] for n in names], fontsize=6, rotation=35, ha='right')
        _style(a, title)
    vals = [100 * models[n]['first_share'] for n in names]
    b = ax[-1].bar(range(len(names)), vals, color=colors)
    ax[-1].bar_label(b, labels=['%.0f %%' % v for v in vals], color=MUTED, fontsize=6, padding=1)
    ax[-1].set_xticks(range(len(names)), [LABEL[n] for n in names], fontsize=6, rotation=35, ha='right')
    _style(ax[-1], 'first of 8 players, % of matches')
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    models, per = load()
    variants_figure(per, D + 'e7_variants.png')
    memory_figure(per, models, D + 'e7_memory.png')
    order = ['random', 'e7_m1_r0', 'dm3_m0_aux', 'dm3_m1_aux', 'dagger_sp_r2', 'ppo_kl_03000k', 'ppo_explore_01001k',
             FINAL, 'ppo_e6d_00503k', FINAL + '_nomem']
    models_figure({LABEL[n]: models[n] for n in order if n in models}, D + 'e7_models.png', teacher=models['teacher'])
    weapons_figure(models, D + 'e7_weapons.png')
    play_figure(models, D + 'e7_play.png')
    print('written', D + 'e7_{variants,memory,models,weapons,play}.png')
