"""Paired frag differences between models of two dm_eval reports on the same plan (matched by variant and seed).

    python scripts/dm/dm_compare_reports.py reports/dm/e4_bc/e4_fresh_phase4.json reports/dm/e4_bc/e4_fresh.json \
        --pairs dm4_m1_sp:dm4_m1_aux dm4_m1_sp:dm3_m1_aux --out reports/dm/e4_bc/e4_phase4_vs_phase3.json
"""
import argparse
import json

from doomagent.evaluate import paired_diff_ci


def load(paths):
    # {model: {(variant, seed): match}} and {model: summary}; a name in a later report keeps the first one's
    per, models = {}, {}
    for p in paths:
        with open(p) as fh:
            r = json.load(fh)
        for n, ms in r['per_match'].items():
            per.setdefault(n, {(m['variant'], m['seed']): m for m in ms})
            models.setdefault(n, r['models'][n])
    return per, models


def paired(per, a, b):
    keys = sorted(set(per[a]) & set(per[b]))
    m = paired_diff_ci([per[a][k]['frags'] for k in keys], [per[b][k]['frags'] for k in keys])
    return dict(mean=m[0], lo=m[1], hi=m[2], matches=len(keys))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('reports', nargs='+')
    ap.add_argument('--pairs', nargs='+', required=True, help='a:b — frags of a minus b')
    ap.add_argument('--out')
    a = ap.parse_args()
    per, models = load(a.reports)
    out = dict(reports=a.reports, pairs={p: paired(per, *p.split(':')) for p in a.pairs},
               models={n: dict(frags=s['frags'], kd=s['kd'], stuck_share=s['stuck_share'],
                               share_of_teacher=s.get('share_of_teacher')) for n, s in models.items()})
    for p, d in out['pairs'].items():
        print(f"{p:28s} {d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]  n={d['matches']}")
    for n, s in out['models'].items():
        sh = s['share_of_teacher']
        print(f"{n:14s} frags {s['frags']['mean']:.2f} [{s['frags']['lo']:.2f}, {s['frags']['hi']:.2f}]  "
              f"stuck {s['stuck_share']['mean']:.3f}  share {'' if sh is None else f'{sh:.0%}'}")
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(out, fh, indent=1)
