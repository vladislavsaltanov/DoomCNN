import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from doomagent.reports import CATEGORICAL, INK, MUTED, SURFACE, _style  # noqa: E402

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


def place_figure(maps, sectors, out_png):
    # 2x4 small maps: mean activation of a unit over the player's position, walls in gray
    import numpy as np
    from doomagent.dm.nav import walls_from_sectors
    walls = walls_from_sectors(sectors) if sectors else []
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6), facecolor=SURFACE)
    for a, (unit, grid, info) in zip(axes.flat, maps):
        ny, nx = grid.shape
        x0, y0, c = info['x0'], info['y0'], info['cell']
        a.imshow(np.ma.masked_invalid(grid), cmap='Blues', origin='lower', interpolation='nearest',
                 extent=(x0, x0 + nx * c, y0, y0 + ny * c))
        for x1, y1, x2, y2 in walls:
            a.plot((x1, x2), (y1, y2), color='#898781', linewidth=0.6)
        a.set_title(f'unit {unit}: {info["spatial_info"]:.0%} of variance by place', color=MUTED, fontsize=8,
                    loc='left')
        a.set_xticks([])
        a.set_yticks([])
        a.set_aspect('equal')
        for s in a.spines.values():
            s.set_visible(False)
    for a in list(axes.flat)[len(maps):]:
        a.axis('off')
    fig.suptitle('Place maps: the 8 units most tied to where the player is (darker = stronger)',
                 color=INK, fontsize=11, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def selectivity_figure(best, strength, names, out_png, cut=0.1):
    # one small 16x32 grid per condition: units for which that condition is the strongest, toned by |r|
    import numpy as np
    from matplotlib.colors import ListedColormap
    fig, axes = plt.subplots(1, len(names), figsize=(2.6 * len(names), 5.2), facecolor=SURFACE)
    blues = plt.get_cmap('Blues')
    for k, (a, name) in enumerate(zip(axes, names)):
        sel = (best == k) & (strength >= cut)
        tone = np.where(sel, strength, np.nan).reshape(32, 16)
        a.imshow(np.zeros((32, 16)), cmap=ListedColormap(['#ecebe7']), interpolation='nearest')
        a.imshow(np.ma.masked_invalid(tone), cmap=blues, vmin=0, vmax=max(float(strength.max()), cut),
                 interpolation='nearest')
        a.set_title(f'{name.replace("_", " ")}\n{int(sel.sum())} units', color=INK, fontsize=9, loc='left')
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    fig.suptitle(f'Which condition each of the 512 units follows best (|Pearson r| >= {cut}; darker = stronger)',
                 color=INK, fontsize=11, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def states_figure(xy, mode, out_png):
    # t-SNE of the states, one small copy per teacher mode: that mode in blue over the rest in gray
    from doomagent.dm.config import MODES
    fig, axes = plt.subplots(1, len(MODES), figsize=(14, 3.9), facecolor=SURFACE)
    for k, (a, name) in enumerate(zip(axes, MODES)):
        on = mode == k
        a.scatter(xy[~on, 0], xy[~on, 1], s=3, color='#d6d5d0', linewidths=0)
        a.scatter(xy[on, 0], xy[on, 1], s=4, color=ENTITY['teacher'], linewidths=0)
        a.set_title(f'{name}: {on.mean():.0%} of steps', color=INK, fontsize=9, loc='left')
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    fig.suptitle("The network's states (t-SNE), by the teacher's mode at that step",
                 color=INK, fontsize=11, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def curves_figure(logs, out_png, teacher=None, confusion_of=None, action_names=None):
    # logs: {name ('m0', 'm1'): path to .log.json}; held-out loss and accuracy by epoch, val frags by
    # checkpoint with the teacher's CI band, and the row-normalized confusion of one run's .live.json
    import json
    import numpy as np
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2), facecolor=SURFACE,
                           gridspec_kw=dict(width_ratios=(1, 1, 1, 1.15)))
    majority = None
    conf = None
    if confusion_of:
        live = json.load(open(confusion_of))
        conf = np.array(live.get('holdout_confusion') or live['confusion'], float)
        majority = conf.sum(1).max() / max(conf.sum(), 1)  # the most frequent teacher action's share
    for name, path in logs.items():
        log = json.load(open(path))['log']
        e = [r['epoch'] for r in log]
        color = ENTITY.get(name, CATEGORICAL[0])
        for a, key in ((ax[0], 'holdout_ce'), (ax[1], 'holdout_acc')):
            ys = [r.get(key) for r in log]
            a.plot(e, ys, color=color, linewidth=2, marker='o', markersize=4, label=name.upper())
            _end_label(a, e[-1], ys[-1], name.upper())
        f = [r['frags'] for r in log if 'frags' in r]
        ef = [r['epoch'] for r in log if 'frags' in r]
        off = -0.4 if name == 'm0' else 0.4
        ax[2].errorbar([x + off for x in ef], [d['mean'] for d in f],
                       yerr=[[d['mean'] - d['lo'] for d in f], [d['hi'] - d['mean'] for d in f]],
                       fmt='o', color=color, capsize=3, markersize=6, label=name.upper())
    if majority is not None:
        ax[1].axhline(majority, color=MUTED, linestyle='--', linewidth=1)
        _end_label(ax[1], ax[1].get_xlim()[0], majority + 0.02, f'most frequent action {majority:.0%}')
    ax[1].axhline(0.1, color=MUTED, linestyle=':', linewidth=1)
    _end_label(ax[1], ax[1].get_xlim()[0], 0.12, 'chance 10%')
    ax[1].set_ylim(0, 1)
    if teacher:
        ax[2].axhspan(teacher['lo'], teacher['hi'], color=ENTITY['teacher'], alpha=0.15, linewidth=0)
        ax[2].axhline(teacher['mean'], color=ENTITY['teacher'], linewidth=1.5)
        _end_label(ax[2], ax[2].get_xlim()[0], teacher['mean'] + 0.3, f'teacher {teacher["mean"]:.2f}')
    ax[2].axhline(0, color='#c9c8c2', linewidth=0.8)
    _style(ax[0], 'Held-out loss (cross-entropy)')
    _style(ax[1], 'Held-out accuracy')
    _style(ax[2], 'Frags per val match (95% CI)')
    for a in ax[:3]:
        _xlabel(a, 'epoch')
        a.legend(fontsize=7, frameon=False, labelcolor=MUTED)
    a = ax[3]
    if conf is not None:
        share = conf / np.maximum(conf.sum(1, keepdims=True), 1)
        a.imshow(share, cmap='Blues', vmin=0, vmax=1)
        n = len(share)
        for i in range(n):
            for j in range(n):
                if share[i, j] >= 0.1:
                    a.text(j, i, f'{share[i, j]:.0%}'[:-1], ha='center', va='center', fontsize=6,
                           color='white' if share[i, j] > 0.5 else INK)
        names = action_names or [str(i) for i in range(n)]
        a.set_xticks(range(n), names, rotation=50, ha='right', fontsize=7, color=MUTED)
        a.set_yticks(range(n), names, fontsize=7, color=MUTED)
        a.tick_params(length=0)
        for s in a.spines.values():
            s.set_visible(False)
        a.set_title('Held-out confusion, % of row', color=INK, fontsize=10, loc='left')
        a.set_ylabel('teacher', color=MUTED, fontsize=8)
        a.set_xlabel('network', color=MUTED, fontsize=8)
    else:
        a.axis('off')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=SURFACE)
    plt.close(fig)
