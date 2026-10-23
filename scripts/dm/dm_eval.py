"""Students and the teacher on the same matches: summary per model, paired frag differences with the teacher,
M1 against M0 (every M1 model with every M0 model).

    python scripts/dm/dm_eval.py --models runs/dm/dm2_m0_r0.pt runs/dm/dm2_m1_r0.pt --plan val --teacher --out reports/dm/e4_val.json
"""
import argparse
import json
import os

from doomagent.dm.config import CONFIRM_SEEDS, DM, fresh_plan, val_plan
from doomagent.dm.evaluate import evaluate_plan, summarize
from doomagent.evaluate import paired_diff_ci


def _frags(results):
    return [r['frags'] for r in sorted(results, key=lambda r: (r['variant'], r['seed']))]


def _ci(m):
    return dict(mean=m[0], lo=m[1], hi=m[2])


def compare(results, kinds):
    # results: {name: [match results]} on the same plan, 'teacher' optional; kinds: {model name: 'm0' | 'm1'}
    teacher = results.get('teacher')
    t_sum = sum(_frags(teacher)) if teacher else None
    models = {}
    for name, res in results.items():
        s = summarize(res)
        s['share_of_teacher'] = (sum(_frags(res)) / t_sum if t_sum else None) if name != 'teacher' else 1.0
        models[name] = s
    vs_teacher = {n: _ci(paired_diff_ci(_frags(r), _frags(teacher)))
                  for n, r in results.items() if teacher and n != 'teacher'}
    m1_vs_m0 = {f'{b} - {a}': _ci(paired_diff_ci(_frags(results[b]), _frags(results[a])))
                for a, ka in kinds.items() if ka == 'm0' for b, kb in kinds.items() if kb == 'm1'}
    return dict(models=models, vs_teacher=vs_teacher, m1_vs_m0=m1_vs_m0)


if __name__ == '__main__':  # worker processes re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--plan', choices=['val', 'fresh', 'confirm'], default='val')
    ap.add_argument('--teacher', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from doomagent.dm.agent import DMChooser
    from doomagent.dm.model import load_model
    from doomagent.dm.teacher import DMTeacher
    from doomagent.variants import BASE
    plan = {'val': val_plan, 'fresh': fresh_plan, 'confirm': lambda: [(BASE, CONFIRM_SEEDS)]}[a.plan]()
    results, kinds = {}, {}
    if a.teacher:
        results['teacher'] = evaluate_plan(plan, DMTeacher(DM), DM, workers=a.workers)
    for path in a.models:
        m = load_model(path, DM)
        name = os.path.splitext(os.path.basename(path))[0]
        kinds[name] = 'm1' if m.memory else 'm0'
        results[name] = evaluate_plan(plan, DMChooser(m.state_dict(), m.memory, DM), DM, workers=a.workers)
        print(name, 'frags', round(summarize(results[name])['frags']['mean'], 2), flush=True)
    out = compare(results, kinds)
    out.update(plan=a.plan, matches=sum(len(s) for _, s in plan), models_paths=a.models,
               per_match={n: [dict(variant=r['variant'], seed=r['seed'], frags=r['frags'], deaths=r['deaths'])
                              for r in res] for n, res in results.items()})
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    for n, s in out['models'].items():
        share = s['share_of_teacher']
        print(n, 'frags %.2f [%.2f, %.2f]' % (s['frags']['mean'], s['frags']['lo'], s['frags']['hi']),
              '' if share is None else 'share of teacher %.2f' % share)
