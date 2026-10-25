"""E7: the final test report (reports/dm/e7_test/): the spec v2 criteria with paired 95% CIs, tables and figures.

    python scripts/dm/dm_e7_report.py

Reads e7_test.json (the teacher, the random player, the students, E6c with the memory zeroed) and e7_test_control.json
(the control: BC M1 on the base graphics only). Base = 40 TEST_SEEDS matches on the base graphics; unfamiliar = 8 matches
on each of the 6 test variants (48), never used in training or in any choice.
"""
import json

import numpy as np

from doomagent.dm.reports import ENTITY, MUTED, SURFACE, _style, plt
from doomagent.evaluate import bootstrap_ci, paired_diff_ci
from doomagent.reports import CATEGORICAL

D = 'reports/dm/e7_test/'
NAMES = [('teacher', 'учитель'), ('random', 'случайный игрок'), ('e7_m1_r0', 'BC только на базовой графике (контроль)'),
         ('dm3_m0_aux', 'BC M0 (без памяти)'), ('dm3_m1_aux', 'BC M1 (E4)'), ('dagger_sp_r2', 'DAgger (E5)'),
         ('ppo_kl_03000k', 'PPO (E6)'), ('ppo_explore_01001k', 'PPO + движение (E6b)'),
         ('ppo_e6c_01001k', 'PPO + случайные старты (E6c)'), ('ppo_e6d_00503k', 'PPO + демонстрации (E6d)'),
         ('ppo_e6c_01001k_nomem', 'E6c с обнулённой памятью')]
FINAL = 'ppo_e6c_01001k'


def load():
    per = {}
    for f in ('e7_test.json', 'e7_test_control.json'):
        try:
            with open(D + f) as fh:
                for n, ms in json.load(fh)['per_match'].items():
                    per.setdefault(n, {(m['variant'], m['seed']): m for m in ms})
        except FileNotFoundError:
            pass
    return per


def frags(per, name, part):
    keys = sorted(k for k in per[name] if (k[0] == 'base') == (part == 'base') or part == 'all')
    return keys, np.array([per[name][k]['frags'] for k in keys])


def ci(x):
    m, lo, hi = bootstrap_ci(x)
    return m, lo, hi


def pair(per, a, b, part):
    keys = sorted(set(frags(per, a, part)[0]) & set(frags(per, b, part)[0]))
    xa = [per[a][k]['frags'] for k in keys]
    xb = [per[b][k]['frags'] for k in keys]
    return paired_diff_ci(xa, xb)


def fmt(t):
    return '%.2f [%.2f, %.2f]' % t


def bars(per, names, out):
    fig, ax = plt.subplots(figsize=(14, 4.6), facecolor=SURFACE)
    x = np.arange(len(names))
    for k, (part, label, color) in enumerate((('base', 'base graphics (40 test matches)', CATEGORICAL[1]),
                                              ('ood', 'unfamiliar test variants (48)', CATEGORICAL[2]))):
        vals = [ci(frags(per, n, part)[1]) for n in names]
        m = [v[0] for v in vals]
        err = [[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]]
        b = ax.bar(x + (k - 0.5) * 0.38, m, 0.38, yerr=err, color=color, label=label, capsize=2, ecolor=MUTED)
        ax.bar_label(b, labels=['%.1f' % v for v in m], color=MUTED, fontsize=7, padding=2)
    ax.set_xticks(x, [n.replace('_', ' ') for n in names], fontsize=7, rotation=25, ha='right')
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    _style(ax, 'Final test: frags per match (95% CI)')
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def ladder(per, out):
    steps = [('e7_m1_r0', 'BC\nbase only'), ('dm3_m1_aux', 'BC\n(E4)'), ('dagger_sp_r2', 'DAgger\n(E5)'),
             ('ppo_kl_03000k', 'PPO\n(E6)'), ('ppo_explore_01001k', 'PPO+move\n(E6b)'), (FINAL, 'PPO+starts\n(E6c)')]
    steps = [s for s in steps if s[0] in per]
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    for part, label, color in (('base', 'base graphics', CATEGORICAL[1]), ('ood', 'unfamiliar graphics', CATEGORICAL[2])):
        t = frags(per, 'teacher', part)[1].mean()
        share = [100 * frags(per, n, part)[1].mean() / t for n, _ in steps]
        ax.plot(range(len(steps)), share, color=color, linewidth=2.5, marker='o', markersize=8, label=label)
        for i, s in enumerate(share):
            ax.annotate('%.0f %%' % s, (i, s), xytext=(0, 9), textcoords='offset points', ha='center', color=MUTED,
                        fontsize=8)
    ax.axhline(100, color=ENTITY['teacher'], linewidth=1, linestyle='--')
    ax.axhline(70, color=MUTED, linewidth=0.8, linestyle=':')
    ax.axhline(60, color=MUTED, linewidth=0.8, linestyle=':')
    ax.set_xticks(range(len(steps)), [lab for _, lab in steps], fontsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    _style(ax, "Final test: share of the teacher's frags after each step (dotted: spec thresholds 70 % and 60 %)")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    per = load()
    names = [n for n, _ in NAMES if n in per]
    t_base, t_ood = frags(per, 'teacher', 'base')[1].mean(), frags(per, 'teacher', 'ood')[1].mean()
    rows = ['| игрок | база (40) | доля учителя | незнакомая графика (48) | доля учителя |', '|---|---|---|---|---|']
    for n, label in NAMES:
        if n not in per:
            continue
        b, o = ci(frags(per, n, 'base')[1]), ci(frags(per, n, 'ood')[1])
        rows.append(f'| {label} (`{n}`) | {fmt(b)} | {100 * b[0] / t_base:.0f} % | {fmt(o)} | {100 * o[0] / t_ood:.0f} % |')
    crit = ['| критерий спека v2 §1 | значение | выполнен |', '|---|---|---|']
    fb = frags(per, FINAL, 'base')[1].mean()
    crit.append(f'| база: лучшая модель ≥ 70 % фрагов учителя | {100 * fb / t_base:.0f} % ({fb:.2f} против {t_base:.2f}) | '
                f'{"да" if fb >= 0.7 * t_base else "нет"} |')
    fo = frags(per, FINAL, 'ood')[1].mean()
    crit.append(f'| незнакомая графика: ≥ 60 % учителя | {100 * fo / t_ood:.0f} % ({fo:.2f} против {t_ood:.2f}) | '
                f'{"да" if fo >= 0.6 * t_ood else "нет"} |')
    if 'e7_m1_r0' in per:
        d = pair(per, FINAL, 'e7_m1_r0', 'ood')
        crit.append(f'| незнакомая графика: значимо больше ученика только на базовой графике | {fmt(d)} | '
                    f'{"да" if d[1] > 0 else "нет"} |')
    if 'dm3_m0_aux' in per and 'dm3_m1_aux' in per:
        d = pair(per, 'dm3_m1_aux', 'dm3_m0_aux', 'all')
        half = (d[2] - d[1]) / 2
        tt = frags(per, 'teacher', 'all')[1].mean()
        crit.append(f'| память: M1 − M0 (BC, равные данные и бюджет), половина интервала ≤ 15 % учителя | {fmt(d)}, '
                    f'половина {half:.2f} при пороге {0.15 * tt:.2f} | {"да" if half <= 0.15 * tt else "нет"} |')
    extra = ['| сравнение (все 88 матчей) | разность фрагов [95 % ДИ] |', '|---|---|']
    for a, b in ((FINAL, 'teacher'), (FINAL, 'dagger_sp_r2'), (FINAL, 'ppo_kl_03000k'), (FINAL, 'ppo_e6d_00503k'),
                 (FINAL, FINAL + '_nomem'), ('dm3_m1_aux', 'dm3_m0_aux')):
        if a in per and b in per:
            extra.append(f'| {a} − {b} | {fmt(pair(per, a, b, "all"))} |')
    bars(per, names, D + 'e7_test.png')
    ladder(per, D + 'e7_ladder.png')
    with open(D + 'e7_test.md', 'w', encoding='utf-8') as fh:
        fh.write('# E7: финальный test\n\n40 матчей на базовой графике (TEST_SEEDS) и по 8 на каждом из 6 test-вариантов '
                 'графики. Ни эти seed, ни эти варианты не использовались ни в обучении, ни при выборе моделей. '
                 'Фраги за 2-минутный матч, 95 % bootstrap-интервал.\n\n## Критерии спека\n\n' + '\n'.join(crit) +
                 '\n\n## Все игроки\n\n' + '\n'.join(rows) + '\n\n## Парные сравнения\n\n' + '\n'.join(extra) +
                 '\n\n![test](e7_test.png)\n\n![лестница](e7_ladder.png)\n')
    print('\n'.join(crit))
    print('\n'.join(rows))
    print('\n'.join(extra))
