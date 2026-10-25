"""Does 160x120 show bots wide enough? Teacher frags when it may not shoot bots narrower than w px,
and the widths of the bots it does shoot.

    python scripts/dm/dm_resolution.py  ->  reports/dm/v1/d1_teacher/d1_resolution.md, reports/dm/v1/d1_teacher/d1_resolution.png
"""
import argparse
import sys
from dataclasses import replace

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from doomagent.dm.config import DM, SHOOT, val_plan  # noqa: E402
from doomagent.dm.evaluate import evaluate_plan, summarize  # noqa: E402
from doomagent.dm.reports import ENTITY, _dots  # noqa: E402
from doomagent.dm.teacher import DMTeacher  # noqa: E402
from doomagent.evaluate import paired_diff_ci  # noqa: E402
from doomagent.reports import MUTED, SURFACE, _style  # noqa: E402

WIDTHS = (0.0, 2.0, 3.0, 4.0, 6.0)
CHECK_W = 4.0 * DM.width / 160  # 16 px at 640: the narrowest bot the network should still be asked to hit


class WidthLog:
    # the teacher, plus the box width of every bot it fires at
    def __init__(self, cfg):
        self.teacher = DMTeacher(cfg)

    def reset(self):
        self.teacher.reset()
        self.widths = []

    def __call__(self, obs, priv):
        a = self.teacher(obs, priv)
        if a == SHOOT:
            self.widths.append(self.teacher.target.width_px)
        return a

    def take_trace(self):
        return dict(widths=list(self.widths))


if __name__ == '__main__':  # worker processes re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default='reports/dm/v1/d1_teacher/d1_resolution')
    a = ap.parse_args()
    plan = val_plan()
    res = {}
    for w in WIDTHS:
        cfg = replace(DM, min_target_w=w)
        res[w] = evaluate_plan(plan, WidthLog(cfg), cfg, workers=a.workers, trace=w == 0.0)
    widths = np.concatenate([r['chooser']['widths'] for r in res[0.0]])
    base = [r['frags'] for r in res[0.0]]
    md = ['# D1: is 160x120 enough? (teacher on the val matches)', '', f'Command: `python {" ".join(sys.argv)}`',
          '', '| no shots at bots narrower than (px at 160) | frags [95% CI] | vs no limit [95% CI] | loss |',
          '| --- | --- | --- | --- |']
    stats = []
    for w in WIDTHS:
        s = summarize(res[w])
        stats.append(s['frags'])
        d = paired_diff_ci([r['frags'] for r in res[w]], base)
        loss = -d[0] / max(np.mean(base), 1e-9)
        md.append(f'| {w:.0f} | {s["frags"]["mean"]:.2f} [{s["frags"]["lo"]:.2f}, {s["frags"]["hi"]:.2f}] | '
                  f'{d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}] | {100 * loss:.0f}% |')
        if w == CHECK_W:
            check = loss
    md += ['', f'Rockets fired: {len(widths)}; bot width at the shot: median {np.median(widths):.1f} px, '
           f'{100 * np.mean(widths < CHECK_W):.0f}% narrower than {CHECK_W:.0f} px.', '',
           (f'Verdict: forbidding shots under {CHECK_W:.0f} px costs {100 * check:.0f}% of the frags: '
            + ('more than 15%, consider 200x150 (a plan change).' if check > 0.15
               else 'within 15%, 160x120 stays.'))]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    _dots(ax[0], [f'{w:.0f} px' for w in WIDTHS], stats, [ENTITY['teacher']] * len(WIDTHS))
    _style(ax[0], 'Teacher frags per match, no shots under w px (95% CI)')
    ax[1].hist(widths, bins=np.arange(0, max(widths.max(), 12) + 1, 1), color=ENTITY['teacher'], rwidth=0.85)
    ax[1].axvline(CHECK_W, color=MUTED, linestyle='--', linewidth=0.8)
    _style(ax[1], 'Bot box width when the teacher fires (px at 160)')
    fig.tight_layout()
    fig.savefig(a.out + '.png', dpi=130, facecolor=SURFACE)
    plt.close(fig)
    with open(a.out + '.md', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(md) + '\n')
    print('\n'.join(md))
