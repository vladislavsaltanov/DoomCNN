import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from doomagent.reports import CATEGORICAL, MUTED, SURFACE, _style  # noqa: E402

# every chart gives an entity the same color (slots of reports.CATEGORICAL; bots stay neutral)
ENTITY = {'teacher': '#2a78d6', 'm0': '#eb6834', 'm1': '#1baf7a', 'random': '#6250d6',
          'bots': '#898781', 'ppo': '#e87ba4', 'ppo_scratch': '#008300'}


def _end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords='offset points', color=MUTED, fontsize=7,
                va='center')


def _xlabel(ax, text):
    ax.set_xlabel(text, color=MUTED, fontsize=8)


def d0_figure(physics, bots, out_png):
    # physics: {'forward': [...], 'strafe_left': [...], 'turn_left': {'hold', 'alternate'},
    # 'fine_left': {...}} per step; bots: {n: {'bot_median': {mean, lo, hi}, 'ms_per_step'}}
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.7), facecolor=SURFACE)
    for i, key in enumerate(('forward', 'strafe_left')):
        ys = physics[key]
        name = key.replace('_', ' ')
        ax[0].plot(range(1, len(ys) + 1), ys, color=CATEGORICAL[i], linewidth=2, marker='o',
                   markersize=4, label=name)
        _end_label(ax[0], len(ys), ys[-1], name)
    _style(ax[0], 'Distance per step, button held (map units)')
    for i, key in enumerate(('turn_left', 'fine_left')):
        for style, mode in (('-', 'hold'), ('--', 'alternate')):
            ys = physics[key][mode]
            ax[1].plot(range(1, len(ys) + 1), ys, color=CATEGORICAL[i], linestyle=style, linewidth=2,
                       marker='o', markersize=3, label=f'{key.replace("_", " ")}, {mode}')
    ax[1].axhline(0, color='#c9c8c2', linewidth=0.8)
    _style(ax[1], 'Turn per step (degrees, + is left)')
    ns = sorted(bots)
    mean = [bots[n]['bot_median']['mean'] for n in ns]
    err = [[m - bots[n]['bot_median']['lo'] for n, m in zip(ns, mean)],
           [bots[n]['bot_median']['hi'] - m for n, m in zip(ns, mean)]]
    ax[2].errorbar(ns, mean, yerr=err, fmt='o', color=ENTITY['bots'], capsize=4, markersize=8)
    ax[2].set_xticks(ns)
    _style(ax[2], "Median bot's frags per match (95% CI)")
    ms = [bots[n]['ms_per_step'] for n in ns]
    bars = ax[3].bar([str(n) for n in ns], ms, color=ENTITY['bots'], width=0.6)
    ax[3].bar_label(bars, labels=['%.1f' % v for v in ms], color=MUTED, fontsize=8, padding=2)
    _style(ax[3], 'Time per agent step (ms)')
    for a in ax[:2]:
        _xlabel(a, 'step')
    ax[0].legend(fontsize=7, frameon=False, labelcolor=MUTED)
    ax[1].legend(fontsize=7, frameon=False, labelcolor=MUTED, ncol=2, loc='upper center',
                 bbox_to_anchor=(0.5, -0.16))  # below the panel: the zigzags fill the plot
    for a in ax[2:]:
        _xlabel(a, 'bots in the match')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _ci(d, f='%.2f'):
    return (f + ' [' + f + ', ' + f + ']') % (d['mean'], d['lo'], d['hi'])


def match_table_md(report):
    # report: {name: dm.evaluate.summarize(...)}
    head = ['player', 'frags [95% CI]', 'deaths', 'K/D (mean, total)', 'suicides', 'hits per rocket',
            'items', 'path', 'stuck', 'sees a bot', "median bot's frags [95% CI]"]
    lines = ['| ' + ' | '.join(head) + ' |', '|' + ' --- |' * len(head)]
    for name, s in report.items():
        lines.append('| ' + ' | '.join([
            name, _ci(s['frags']), '%.1f' % s['deaths']['mean'],
            '%.2f, %.2f' % (s['kd']['mean'], s['kd_total']), '%.1f' % s['suicides']['mean'],
            '%.2f' % s['hits_per_shot']['mean'], '%.1f' % s['items']['mean'],
            '%.0f' % s['path']['mean'], '%.1f%%' % (100 * s['stuck_share']['mean']),
            '%.0f%%' % (100 * s['enemy_share']['mean']),
            _ci(s['bot_median']) if 'bot_median' in s else '-']) + ' |')
    return '\n'.join(lines) + '\n'


def gate_a(report):
    # spec D1: K/D > 1, frags not below the median bot of the same matches, stuck under 5 %
    t = report['teacher']
    return [('K/D over all matches > 1', t['kd_total'], t['kd_total'] > 1.0),
            ("frags >= median bot's frags", (t['frags']['mean'], t['bot_median']['mean']),
             t['frags']['mean'] >= t['bot_median']['mean']),
            ('stuck share < 5 %', t['stuck_share']['mean'], t['stuck_share']['mean'] < 0.05)]


def _dots(ax, names, stats, colors):
    m = [s['mean'] for s in stats]
    err = [[s['mean'] - s['lo'] for s in stats], [s['hi'] - s['mean'] for s in stats]]
    x = range(len(names))
    for i in x:
        ax.errorbar([i], [m[i]], yerr=[[err[0][i]], [err[1][i]]], fmt='o', color=colors[i],
                    capsize=4, markersize=8)
        ax.annotate('%.2f' % m[i], (i, m[i]), xytext=(8, 0), textcoords='offset points',
                    color=MUTED, fontsize=8, va='center')
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.axhline(0, color='#c9c8c2', linewidth=0.8)


def teacher_figure(report, mode_share, out_png):
    t, r = report['teacher'], report.get('random')
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), facecolor=SURFACE)
    names, stats, colors = ['teacher', 'median bot'], [t['frags'], t['bot_median']], \
        [ENTITY['teacher'], ENTITY['bots']]
    if r is not None:
        names, stats, colors = names + ['random'], stats + [r['frags']], colors + [ENTITY['random']]
    _dots(ax[0], names, stats, colors)
    _style(ax[0], 'Frags per match (95% CI)')
    kd_names = ['teacher'] + (['random'] if r is not None else [])
    _dots(ax[1], kd_names, [t['kd']] + ([r['kd']] if r is not None else []),
          [ENTITY['teacher']] + ([ENTITY['random']] if r is not None else []))
    ax[1].axhline(1.0, color=MUTED, linewidth=0.8, linestyle='--')
    _style(ax[1], 'K/D per match (95% CI); dashed: Gate A')
    modes = list(mode_share)
    bars = ax[2].bar(modes, [mode_share[m] for m in modes], color=ENTITY['teacher'], width=0.6)
    ax[2].bar_label(bars, labels=['%.0f%%' % (100 * mode_share[m]) for m in modes], color=MUTED,
                    fontsize=8, padding=2)
    _style(ax[2], "Teacher's steps by decision")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=SURFACE)
    plt.close(fig)
