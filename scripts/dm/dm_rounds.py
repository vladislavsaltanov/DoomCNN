"""E5: DAgger rounds for one arm. Round r records dagger_plan(r) with the best model so far playing (beta 0.3 in
round 1, then 0; the teacher labels every step and the recorder writes the targets of every head on the student's
states), then trains from that model on R0, R1 and all rounds of the arm (3 epochs, lr 1e-4, val every epoch) and
applies stop_rule. Finished recordings and trainings are skipped, so the script can be restarted.

    python scripts/dm/dm_rounds.py --tag aux --start runs/dm/dm3_m1_aux.pt
    python scripts/dm/dm_rounds.py --tag sp --start runs/dm/dm3_m1_aux.pt --spatial 1.0 --first-data data/dm/dagger_aux_r1
    python scripts/dm/dm_rounds.py --tag sp_w5 --start runs/dm/dagger_sp_r2.pt --spatial 1.0 --dagger-weight 5         --prior-data data/dm/dagger_aux_r1 data/dm/dagger_sp_r2 --first-round 3

Every round's model then plays the fresh seeds (reports/dm/e5_fresh_<tag>.json): the val choice is noisy.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from dm_runs import R01, finished  # noqa: E402

from doomagent.dm.dagger import stop_rule  # noqa: E402

TEACHER_VAL = 11.90  # the final teacher on val (Gate A v2.1 recount)


def best_score(model):
    with open(model + '.log.json') as fh:
        return max(r['val_score'] for r in json.load(fh)['log'] if 'val_score' in r)


def run(cmd, log):
    print(time.strftime('%H:%M:%S'), ' '.join(cmd[2:5]), '->', log, flush=True)
    with open(log, 'w') as fh:
        code = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if code:
        sys.exit(f'failed ({code}): see {log}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--kind', choices=['m0', 'm1'], default='m1')
    ap.add_argument('--aux', type=float, default=1.0)
    ap.add_argument('--spatial', type=float, default=0.0)
    ap.add_argument('--first-data', help='round 1 data of another arm (the same start model played it)')
    ap.add_argument('--prior-data', nargs='*', default=[], help='rounds recorded before, trained on with the new ones')
    ap.add_argument('--first-round', type=int, default=1, help='dagger_plan round of the first new round (its seeds)')
    ap.add_argument('--max-rounds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--dagger-weight', type=float, default=1.0,
                    help='windows of the round data drawn this many times as often')
    a = ap.parse_args()
    py = [sys.executable, '-u']
    models, scores, dirs = [a.start], [best_score(a.start)], list(a.prior_data)
    state = f'runs/dm/dagger_{a.tag}.json'
    for r in range(a.first_round, a.first_round + a.max_rounds):
        best = models[stop_rule(scores, TEACHER_VAL)[1]]
        data = a.first_data if r == a.first_round and a.first_data else f'data/dm/dagger_{a.tag}_r{r}'
        if not os.path.exists(os.path.join(data, 'meta.json')):
            run(py + ['-m', 'doomagent.dm.dagger', '--model', best, '--out', data, '--round', str(r),
                      '--beta', '0.3' if r == 1 else '0'], data.rstrip('/') + '.out')
        dirs.append(data)
        out = f'runs/dm/dagger_{a.tag}_r{r}.pt'
        if not finished(out, a.epochs):
            run(py + ['-m', 'doomagent.dm.train', a.kind, *R01, *dirs, '--out', out, '--epochs', str(a.epochs),
                      '--lr', '1e-4', '--eval-every', '1', '--init', best, '--aux', str(a.aux),
                      '--spatial', str(a.spatial)] +
                [x for d in dirs if a.dagger_weight != 1 for x in ('--weight', f'{d}={a.dagger_weight}')],
                out[:-3] + '.out')
        models.append(out)
        scores.append(best_score(out))
        stop, b = stop_rule(scores, TEACHER_VAL)
        with open(state, 'w') as fh:
            json.dump(dict(tag=a.tag, start=a.start, spatial=a.spatial, models=models, scores=scores, data=dirs,
                           best=models[b], stopped=stop), fh, indent=1)
        print(time.strftime('%H:%M:%S'), f'round {r}: val score {scores[-1]:.2f}, best {models[b]}', flush=True)
        if stop:
            break
    # every round's model on the fresh seeds, paired later with e4_fresh.json (the same matches)
    fresh = f'reports/dm/e5_fresh_{a.tag}.json'
    if not os.path.exists(fresh):
        run(py + ['scripts/dm/dm_eval.py', '--models', *models[1:], '--plan', 'fresh', '--workers', '4',
                  '--out', fresh],
            f'runs/dm/dagger_{a.tag}_fresh.out')
