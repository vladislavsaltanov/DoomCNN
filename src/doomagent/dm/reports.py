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
