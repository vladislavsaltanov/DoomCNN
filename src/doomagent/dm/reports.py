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


def e0_figure(turns, moves, bots, out_png):
    # turns: {requested degrees: executed}; moves: {move name: map units per step, steady}; bots: {'player',
    # 'bot_median', 'bot_best': {mean, lo, hi}} frags per match with a random player
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.8), facecolor=SURFACE, gridspec_kw=dict(width_ratios=(1.4, 1.2, 0.8)))
    req = sorted(turns)
    err = [turns[d] - d for d in req]
    bars = ax[0].bar([f'{d:+g}' for d in req], err, color=CATEGORICAL[0], width=0.6)
    ax[0].bar_label(bars, labels=['%+.2f' % e for e in err], color=MUTED, fontsize=7, padding=2)
    ax[0].axhline(0, color='#c9c8c2', linewidth=0.8)
    _style(ax[0], 'Turn error: executed minus requested (degrees)')
    _xlabel(ax[0], 'requested turn per step, degrees (+ is left)')
    names = list(moves)
    bars = ax[1].bar([n.replace('_', ' ') for n in names], [moves[n] for n in names], color=CATEGORICAL[0], width=0.6)
    ax[1].bar_label(bars, labels=['%.1f' % moves[n] for n in names], color=MUTED, fontsize=7, padding=2)
    ax[1].tick_params(axis='x', labelrotation=30)
    _style(ax[1], 'Distance per step at full speed (map units)')
    keys = [('player', 'random player', ENTITY['random']), ('bot_median', 'median bot', ENTITY['bots']),
            ('bot_best', 'best bot', ENTITY['bots'])]
    for i, (k, name, color) in enumerate(keys):
        d = bots[k]
        ax[2].errorbar([i], [d['mean']], yerr=[[d['mean'] - d['lo']], [d['hi'] - d['mean']]], fmt='o', color=color,
                       capsize=4, markersize=8)
        _end_label(ax[2], i, d['mean'], '%.1f' % d['mean'])
    ax[2].set_xticks(range(len(keys)), [n for _, n, _ in keys])
    ax[2].set_xlim(-0.5, len(keys) - 0.3)
    ax[2].axhline(0, color='#c9c8c2', linewidth=0.8)
    _style(ax[2], 'Frags per 2-minute match (95% CI)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _ci(d, f='%.2f'):
    return (f + ' [' + f + ', ' + f + ']') % (d['mean'], d['lo'], d['hi'])


def match_table_md(report):
    # report: {name: dm.evaluate.summarize(...)}
    head = ['player', 'frags [95% CI]', 'deaths', 'K/D (mean, total)', 'suicides', 'hits per shot',
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


def gate_a2(report, diff_ci):
    # spec v2 §1, revised 2026-10-23 (v2.1, user decision on the E1 numbers): K/D over all matches >= 3, first of 8
    # in >= 50 % of matches, frags not significantly below the best bot of the match: diff_ci = (mean, lo, hi) of the
    # per-match difference teacher - best bot, its upper bound >= 0
    t = report['teacher']
    return [('K/D over all matches >= 3', t['kd_total'], t['kd_total'] >= 3.0),
            ('first of 8 in >= 50 % of matches', t['first_share'], t['first_share'] >= 0.5),
            ('frags - best bot, upper 95% bound >= 0', tuple(round(x, 2) for x in diff_ci), diff_ci[2] >= 0.0)]


def teacher_figure2(report, diag, out_png):
    # spec v2: frags against the best and the median bot, the weapon in hand, death causes, hits by weapon;
    # diag: {'death_cause': {cause: share}, 'hit_share_by_weapon': {weapon: hits per shot}}
    from doomagent.dm.weapons import WEAPONS
    t = report['teacher']
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.9), facecolor=SURFACE, gridspec_kw=dict(width_ratios=(1, 1.3, 1, 1.2)))
    _dots(ax[0], ['teacher', 'best bot', 'median bot'], [t['frags'], t['bot_best'], t['bot_median']],
          [ENTITY['teacher'], ENTITY['bots'], ENTITY['bots']])
    ax[0].axhline(2.0 * t['bot_best']['mean'], color=MUTED, linewidth=0.8, linestyle='--')
    _style(ax[0], 'Frags per match (95% CI); dashed: 2 x best bot')
    for a, data, title, fmt in (
            (ax[1], {WEAPONS[k].name: v for k, v in t['weapon_share'].items()}, 'Weapon in hand (share of steps)', '%.0f%%'),
            (ax[2], diag['death_cause'], 'Deaths by cause (share)', '%.0f%%'),
            (ax[3], diag['hit_share_by_weapon'], 'Hits per shot by weapon', '%.2f')):
        names = list(data)
        vals = [data[n] for n in names]
        bars = a.bar(range(len(names)), vals, color=ENTITY['teacher'], width=0.65)
        a.bar_label(bars, labels=[fmt % (100 * v if fmt.endswith('%%') else v) for v in vals], color=MUTED,
                    fontsize=7, padding=2)
        a.set_xticks(range(len(names)), names, rotation=30, ha='right', fontsize=7)
        _style(a, title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=SURFACE)
    plt.close(fig)


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


def models_figure(models, out_png, teacher=None):
    # models: {name: dm.evaluate.summarize(...)} of students on the same matches; frags, K/D and the stuck share with
    # 95% CIs; with the teacher's summary, dotted lines at Gate B (half its frags) and at its stuck share
    names = list(models)
    colors = [CATEGORICAL[1 + i % (len(CATEGORICAL) - 1)] for i in range(len(names))]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.2), facecolor=SURFACE)
    panels = ((ax[0], 'frags', 'Frags per match (95% CI); dotted: Gate B', 0.5),
              (ax[1], 'kd', 'K/D per match (95% CI)', None),
              (ax[2], 'stuck_share', 'Share of steps stuck (95% CI); dotted: teacher', 1.0))
    for a, key, title, k in panels:
        _dots(a, names, [models[n][key] for n in names], colors)
        a.tick_params(axis='x', labelrotation=30)
        if teacher is not None and k is not None:
            a.axhline(k * teacher[key]['mean'], color=ENTITY['teacher'], linewidth=0.8, linestyle=':')
        _style(a, title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _smooth(x, k):
    import numpy as np
    x = np.asarray(x, float)
    return x if len(x) < k else np.convolve(x, np.ones(k) / k, mode='valid')


def ppo_figure(rows, out_png, teacher_val=None, smooth=20):
    # rows: dm.ppo log.jsonl. Val score by env step (dashed: the start, dotted: Gate B), the training matches'
    # frags and stuck share, agreement with the teacher by head, the losses, and PPO's KL, clipped share, entropy
    import numpy as np
    from doomagent.dm.actions import HEADS
    ev = [r for r in rows if r['kind'] == 'eval']
    up = [r for r in rows if r['kind'] == 'update' and r['phase'] == 'ppo']
    um = [r for r in rows if r['kind'] == 'update' and 'm_frags' in r]
    fig, ax = plt.subplots(2, 3, figsize=(18, 8), facecolor=SURFACE)
    ax = ax.ravel()
    M = 1e6

    def line(a, xs, ys, name, color, k=1, **kw):
        if len(ys):
            ys = _smooth(ys, k)
            xs = np.asarray(xs[len(xs) - len(ys):]) / M
            a.plot(xs, ys, color=color, linewidth=2, **kw)
            _end_label(a, xs[-1], ys[-1], name)

    if ev:
        x = np.array([r['step'] for r in ev]) / M
        f = [r['frags'] for r in ev]
        ax[0].errorbar(x, [d['mean'] for d in f], yerr=[[d['mean'] - d['lo'] for d in f], [d['hi'] - d['mean'] for d in f]],
                       fmt='o', color=ENTITY['ppo'], capsize=3, markersize=4)
        line(ax[0], [r['step'] for r in ev], [r['val_score'] for r in ev], 'val score', ENTITY['m1'])
        ax[0].axhline(ev[0]['val_score'], color=MUTED, linewidth=0.8, linestyle='--')
        if teacher_val:
            ax[0].axhline(0.5 * teacher_val, color=ENTITY['teacher'], linewidth=0.8, linestyle=':')
    _style(ax[0], 'Val: frags (95% CI) and val score; dashed: start, dotted: Gate B')
    line(ax[1], [r['step'] for r in um], [r['m_frags'] for r in um], 'frags', ENTITY['ppo'], smooth)
    line(ax[1], [r['step'] for r in um], [r['m_deaths'] for r in um], 'deaths', ENTITY['bots'], smooth)
    _style(ax[1], f'Training matches (sampled policy), mean of {smooth} updates')
    line(ax[2], [r['step'] for r in um], [r['m_stuck_share'] for r in um], 'training', ENTITY['ppo'], smooth)
    if ev:
        line(ax[2], [r['step'] for r in ev], [r['stuck_share'] for r in ev], 'val', ENTITY['m1'], marker='o')
    _style(ax[2], 'Share of steps stuck')
    for i, h in enumerate(HEADS + ('',)):
        key = f'agree_{h}' if h else 'agree'
        line(ax[3], [r['step'] for r in up], [r[key] for r in up], h or 'all four', CATEGORICAL[1 + i % 6], smooth)
    _style(ax[3], "Policy's argmax = teacher's label, on its own states")
    for i, (k, name) in enumerate((('ks', 'teacher CE'), ('aux', 'auxiliary'), ('spatial', 'spatial'), ('v_loss', 'value'))):
        line(ax[4], [r['step'] for r in up], [r[k] for r in up], name, CATEGORICAL[1 + i], smooth)
    _style(ax[4], 'Losses')
    for i, (k, name) in enumerate((('kl', 'KL x 10'), ('clipfrac', 'clipped'), ('ent_turn', 'entropy turn'),
                                   ('ent_move', 'entropy move'), ('ks_w', 'teacher weight'))):
        ys = [r[k] * (10 if k == 'kl' else 1) for r in up]
        line(ax[5], [r['step'] for r in up], ys, name, CATEGORICAL[1 + i], 1 if k == 'ks_w' else smooth)
    _style(ax[5], 'PPO')
    for a in ax:
        _xlabel(a, 'env steps, millions')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=SURFACE)
    plt.close(fig)


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
    fig.subplots_adjust(top=0.84)  # room for the suptitle above the panel titles
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


def curves_figure(logs, out_png, teacher=None, confusion_of=None, head='turn'):
    # logs: {name ('m0', 'm1'): path to .log.json}. Held-out loss (sum of the heads) and accuracy of each head by
    # epoch (small multiples, one panel per head), val frags by checkpoint with the teacher's CI band, and the
    # row-normalized held-out confusion of one head of one run's .live.json
    import json
    import numpy as np
    from doomagent.dm.actions import HEADS, MOVE_NAMES, TURN_DEG
    values = dict(move=list(MOVE_NAMES), turn=[f'{d:+g}' for d in TURN_DEG], fire=['no', 'fire'], weapon=['keep', 'next'])
    fig = plt.figure(figsize=(20, 7.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, height_ratios=(1, 1))
    ax_ce, ax_val = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_heads = [fig.add_subplot(gs[1, k]) for k in range(4)]
    ax_conf = fig.add_subplot(gs[0, 2:])
    conf = None
    if confusion_of:
        live = json.load(open(confusion_of))
        conf = np.array((live.get('holdout_confusion') or live['confusion'])[head], float)
    for i, (name, path) in enumerate(logs.items()):
        log = json.load(open(path))['log']
        e = [r['epoch'] for r in log]
        color = ENTITY.get(name, CATEGORICAL[1 + i % (len(CATEGORICAL) - 1)])  # slot 0 is the teacher's
        ax_ce.plot(e, [r.get('holdout_ce') for r in log], color=color, linewidth=2, marker='o', markersize=4,
                   label=name.upper())
        for a, h in zip(ax_heads, HEADS):
            ys = [r.get(f'holdout_acc_{h}') for r in log]
            a.plot(e, ys, color=color, linewidth=2, marker='o', markersize=4, label=name.upper())
            _end_label(a, e[-1], ys[-1], name.upper())
        f = [r['frags'] for r in log if 'frags' in r]
        ef = [r['epoch'] for r in log if 'frags' in r]
        off = {'m0': -0.4, 'm1': 0.4}.get(name, (i - (len(logs) - 1) / 2) * 0.2)
        ax_val.errorbar([x + off for x in ef], [d['mean'] for d in f],
                        yerr=[[d['mean'] - d['lo'] for d in f], [d['hi'] - d['mean'] for d in f]],
                        fmt='o', color=color, capsize=3, markersize=6, label=name.upper())
    for a, h in zip(ax_heads, HEADS):
        a.set_ylim(0, 1)
        _style(a, f'Held-out accuracy: {h}')
        _xlabel(a, 'epoch')
    if teacher:
        ax_val.axhspan(teacher['lo'], teacher['hi'], color=ENTITY['teacher'], alpha=0.15, linewidth=0)
        ax_val.axhline(teacher['mean'], color=ENTITY['teacher'], linewidth=1.5)
        _end_label(ax_val, ax_val.get_xlim()[0], teacher['mean'] + 0.3, f'teacher {teacher["mean"]:.2f}')
    ax_val.axhline(0, color='#c9c8c2', linewidth=0.8)
    _style(ax_ce, 'Held-out loss (sum of the four heads)')
    _style(ax_val, 'Frags per val match (95% CI)')
    for a in (ax_ce, ax_val):
        _xlabel(a, 'epoch')
        a.legend(fontsize=7, frameon=False, labelcolor=MUTED)
    if conf is not None:
        share = conf / np.maximum(conf.sum(1, keepdims=True), 1)
        ax_conf.imshow(share, cmap='Blues', vmin=0, vmax=1)
        n = len(share)
        for i in range(n):
            for j in range(n):
                if share[i, j] >= 0.1:
                    ax_conf.text(j, i, f'{share[i, j]:.0%}'[:-1], ha='center', va='center', fontsize=6,
                                 color='white' if share[i, j] > 0.5 else INK)
        ax_conf.set_xticks(range(n), values[head], rotation=50, ha='right', fontsize=7, color=MUTED)
        ax_conf.set_yticks(range(n), values[head], fontsize=7, color=MUTED)
        ax_conf.tick_params(length=0)
        for sp in ax_conf.spines.values():
            sp.set_visible(False)
        ax_conf.set_title(f'Held-out confusion of the {head} head, % of row', color=INK, fontsize=10, loc='left')
        ax_conf.set_ylabel('teacher', color=MUTED, fontsize=8)
        ax_conf.set_xlabel('network', color=MUTED, fontsize=8)
    else:
        ax_conf.axis('off')
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def dataset_stats(dirs):
    # what the recorded teacher matches hold: size, lives, label shares per head, modes, noise, weapon, frags
    import glob
    import os
    from collections import Counter

    import numpy as np

    from doomagent.dm.actions import HEAD_SIZES, HEADS, N_WEAPON_SLOTS
    from doomagent.dm.config import MODES
    files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, 'ep_*.npz')))
    heads = {h: np.zeros(k) for h, k in zip(HEADS, HEAD_SIZES)}
    modes, weapons = np.zeros(len(MODES)), np.zeros(N_WEAPON_SLOTS + 1)
    steps, lives, frags, noisy, size, per = [], [], [], 0, 0, Counter()
    for f in files:
        z = np.load(f)
        a = z['actions']
        steps.append(len(a))
        lives.append(int(z['first'].sum()))
        frags.append(float(z['rewards'].sum()))
        for k, h in enumerate(HEADS):
            heads[h] += np.bincount(a[:, k], minlength=HEAD_SIZES[k])
        modes += np.bincount(z['modes'], minlength=len(MODES))
        weapons += np.bincount(np.clip(z['weapon'], 0, N_WEAPON_SLOTS), minlength=N_WEAPON_SLOTS + 1)
        noisy += int((z['executed'] != a).any(1).sum())
        per[str(z['variant'])] += 1
        size += os.path.getsize(f) + os.path.getsize(f[:-4] + '.frames.npy')
    n = max(sum(steps), 1)
    return dict(matches=len(files), steps=int(sum(steps)), gb=size / 2 ** 30,
                steps_per_match=float(np.mean(steps)) if steps else 0.0,
                lives_per_match=lives, frags_per_match=frags,
                action_share={h: (v / n).tolist() for h, v in heads.items()}, mode_share=(modes / n).tolist(),
                weapon_share={k: float(weapons[k] / n) for k in range(1, N_WEAPON_SLOTS + 1)},
                noise_share=noisy / n, per_variant=dict(per))


def dataset_figure(stats, out_png):
    import numpy as np

    from doomagent.dm.actions import MOVE_NAMES, TURN_DEG
    from doomagent.dm.config import MODES
    from doomagent.dm.weapons import WEAPONS
    color = ENTITY['teacher']
    fig, ax = plt.subplots(2, 3, figsize=(18, 7.6), facecolor=SURFACE)
    bars = ((ax[0, 0], [n.replace('_', ' ') for n in MOVE_NAMES], stats['action_share']['move'], "Teacher's move labels"),
            (ax[0, 1], [f'{d:+g}' for d in TURN_DEG], stats['action_share']['turn'], "Teacher's turn labels (degrees)"),
            (ax[0, 2], list(MODES), stats['mode_share'], "Steps by teacher's mode"),
            (ax[1, 0], [WEAPONS[k].name for k in sorted(stats['weapon_share'])],
             [stats['weapon_share'][k] for k in sorted(stats['weapon_share'])], 'Weapon in hand'))
    for a, names, share, title in bars:
        b = a.bar(range(len(names)), share, color=color, width=0.7)
        a.bar_label(b, labels=[f'{v:.0%}' for v in share], color=MUTED, fontsize=7, padding=2)
        a.set_xticks(range(len(names)), names, rotation=35, ha='right', fontsize=7)
        a.set_yticks([])
        _style(a, title)
    for a, values, title, unit in ((ax[1, 1], stats['lives_per_match'], 'Lives per match', 'lives'),
                                   (ax[1, 2], stats['frags_per_match'], "Teacher's frags per match", 'frags')):
        lo, hi = int(np.floor(min(values))), int(np.ceil(max(values)))
        a.hist(values, bins=np.arange(lo - 0.5, hi + 1.5), color=color, edgecolor=SURFACE, linewidth=2)
        a.yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # whole matches
        a.axvline(np.mean(values), color=INK, linewidth=1, linestyle='--')
        _end_label(a, np.mean(values), a.get_ylim()[1] * 0.92, f'mean {np.mean(values):.1f}')
        _xlabel(a, unit)
        _style(a, title)
    fire = stats['action_share']['fire'][1]
    fig.suptitle(f'{stats["matches"]} matches, {stats["steps"]:,} steps, {stats["gb"]:.1f} GB; fire on {fire:.0%} of steps; '
                 f'{stats["noise_share"]:.1%} of steps played a noise action'.replace(',', ' '),
                 color=INK, fontsize=11, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, facecolor=SURFACE)
    plt.close(fig)

