import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from doomagent.config import ACTION_NAMES, N_ACTIONS  # noqa: E402


def report_table_md(report):
    head = ['model', 'reward mean [95% CI]', 'vs teacher [95% CI]', 'kills/shot',
            'shots', 'presses', 'idle rate', 'length', 'latency ms']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + ' --- |' * len(head)]
    for name, s in report.items():
        r = s['reward']
        vs = s.get('vs_teacher')
        lat = s.get('latency_ms')
        cells = [
            name,
            '%.2f [%.2f, %.2f]' % (r['mean'], r['lo'], r['hi']),
            '-' if vs is None else '%.2f [%.2f, %.2f]' % tuple(vs),
            '%.3f' % s['kills_per_shot'],
            '%.1f' % s['shots']['mean'],
            '%.1f' % s['pressed_per_ep'],
            '%.2f' % s['idle_rate'],
            '%.0f' % s['length']['mean'],
            '-' if lat is None else '%.1f' % lat,
        ]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


# static figures: one series per panel, so one hue and no legend; ink and grid stay recessive
SERIES, SURFACE, INK, MUTED = '#2a78d6', '#fcfcfb', '#0b0b0b', '#52514e'


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc='left')
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis='y', color='#e4e3df', linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#c9c8c2')


def _episodes(dirs):
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, 'ep_*.npz'))):
            yield p[:-4], np.load(p)


def dataset_stats(dirs):
    acts, vis, valid, offs, lens, variants = [], [], [], [], [], {}
    for _, z in _episodes(dirs):
        acts.append(z['actions'])
        vis.append(z['vis'])
        valid.append(z['valid'])
        offs.append(z['off'][z['vis'] == 1])
        lens.append(len(z['actions']))
        v = str(z['variant'])
        variants[v] = variants.get(v, 0) + 1
    acts, vis, valid = np.concatenate(acts), np.concatenate(vis), np.concatenate(valid)
    return dict(episodes=len(lens), steps=int(acts.size), mean_len=float(np.mean(lens)),
                action_share=[float((acts == i).mean()) for i in range(N_ACTIONS)],
                action_share_valid=[float((acts[valid] == i).mean()) for i in range(N_ACTIONS)],
                valid_share=float(valid.mean()), vis_share=float(vis.mean()),
                variants=variants, offsets=np.concatenate(offs), lens=np.array(lens))


def dataset_report(dirs, out_png, out_md):
    s = dataset_stats(dirs)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), facecolor=SURFACE)
    bars = ax[0].bar(ACTION_NAMES, s['action_share_valid'], color=SERIES, width=0.6)
    ax[0].bar_label(bars, labels=['%.1f%%' % (100 * v) for v in s['action_share_valid']],
                    color=MUTED, fontsize=8, padding=2)
    _style(ax[0], 'Teacher label share (steps with ammo)')
    ax[1].hist(s['offsets'], bins=40, range=(-1, 1), color=SERIES, rwidth=0.85)
    _style(ax[1], 'Nearest enemy offset (enemy visible), -1 = left edge')
    ax[2].hist(s['lens'], bins=20, color=SERIES, rwidth=0.85)
    _style(ax[2], 'Episode length (steps)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    shares = ' / '.join('%.3f' % v for v in s['action_share'])
    shares_valid = ' / '.join('%.3f' % v for v in s['action_share_valid'])
    md = ['# Dataset report: ' + ', '.join(dirs), '',
          '- episodes: %d, steps: %d, mean episode length: %.1f'
          % (s['episodes'], s['steps'], s['mean_len']),
          '- action share (%s): %s' % (' / '.join(ACTION_NAMES), shares),
          '- action share on steps with ammo, which the action loss sees: ' + shares_valid,
          '- share of steps with ammo (valid): %.3f' % s['valid_share'],
          '- share of steps with a visible enemy: %.3f' % s['vis_share'],
          '- episodes per variant: ' + json.dumps(s['variants'])]
    with open(out_md, 'w') as fh:
        fh.write('\n'.join(md) + '\n')
    return s


def mask_preview(dirs, out_png, n=4):
    # frames with a visible enemy next to their masks: the mask must sit on the enemy
    tiles = []
    for pre, z in _episodes(dirs):
        frames = np.load(pre + '.frames.npy', mmap_mode='r')
        masks = np.load(pre + '.masks.npy', mmap_mode='r')
        for t in np.flatnonzero(z['vis'] == 1)[::25]:
            m = np.unpackbits(masks[t], axis=-1).astype(bool)
            over = np.repeat(frames[t][..., None], 3, axis=2).copy()
            over[m] = (255, 0, 0)
            tiles.append(np.concatenate([np.repeat(frames[t][..., None], 3, axis=2), over], 1))
            if len(tiles) >= n:
                break
        if len(tiles) >= n:
            break
    if not tiles:
        return
    fig, ax = plt.subplots(len(tiles), 1, figsize=(10, 3.2 * len(tiles)), squeeze=False)
    for a, tile in zip(ax[:, 0], tiles):
        a.imshow(tile, interpolation='nearest')
        a.axis('off')
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
