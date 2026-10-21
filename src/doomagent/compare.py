import argparse
import json
import os
import time

import numpy as np

from doomagent.config import Cfg


def latency_ms(act, cfg: Cfg = Cfg(), n=200):
    obs = np.zeros((cfg.stack, cfg.height, cfg.width), np.uint8)
    for _ in range(20):
        act(obs)
    t = time.perf_counter()
    for _ in range(n):
        act(obs)
    return (time.perf_counter() - t) / n * 1000.0


def budget(data_dirs, rl_steps):
    n = 0
    for d in data_dirs:
        with open(os.path.join(d, 'meta.json')) as fh:
            n += json.load(fh)['steps']
    return dict(data_steps=n, rl_steps=rl_steps, total_steps=n + rl_steps)


def parse_pairs(items):
    return dict(x.split('=', 1) for x in items or [])


def chooser_for(path, cfg: Cfg = Cfg()):
    # a picklable chooser for worker processes: a .pt student, or a PPO .zip whose actor is
    # copied into a StudentCNN (same layers; the auxiliary heads do not choose actions)
    import torch

    from doomagent.agent import StudentChooser
    from doomagent.model import StudentCNN
    if path.endswith('.zip'):
        from stable_baselines3 import PPO
        from doomagent.ppo import policy_state
        state = policy_state(PPO.load(path, device='cpu'), StudentCNN(cfg).state_dict())
    else:
        state = torch.load(path, map_location='cpu')
    return StudentChooser(state, cfg)


def run_report(agent_paths, out_path, baseline='bc0', cfg: Cfg = Cfg(), workers=1):
    from doomagent.agent import load_agent
    from doomagent.evaluate import evaluate_plan, paired_diff_ci, pixel_only, summarize
    from doomagent.teacher import Teacher
    from doomagent.variants import test_plans
    agents = {n: load_agent(p, cfg) for n, p in agent_paths.items()}
    choosers = {'teacher': Teacher(cfg)}
    if workers > 1:
        choosers.update({n: chooser_for(p, cfg) for n, p in agent_paths.items()})
    else:
        choosers.update({n: pixel_only(a.act) for n, a in agents.items()})
    report = {}
    for split, plan in test_plans().items():
        if not plan:
            continue
        results = {n: evaluate_plan(plan, c, cfg, workers=workers) for n, c in choosers.items()}
        rewards = {n: [r['reward'] for r in res] for n, res in results.items()}
        for name, res in results.items():
            s = summarize(res)
            if name != 'teacher':
                s['vs_teacher'] = paired_diff_ci(rewards[name], rewards['teacher'])
                if baseline in rewards and name != baseline:
                    s[f'vs_{baseline}'] = paired_diff_ci(rewards[name], rewards[baseline])
                s['latency_ms'] = latency_ms(agents[name].act, cfg)
            report[f'{name}@{split}'] = s
            print(f'{name}@{split}', json.dumps(s))
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(report, fh, indent=2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('report')
    r.add_argument('--agent', action='append', help='name=path (.pt student or .zip PPO)')
    r.add_argument('--baseline', default='bc0')
    r.add_argument('--out', default='results/test_report.json')
    r.add_argument('--workers', type=int, default=1, help='processes for the episodes')
    b = sub.add_parser('budget')
    b.add_argument('--data', nargs='+', required=True)
    b.add_argument('--rl-steps', type=int, required=True)
    a = ap.parse_args()
    if a.cmd == 'report':
        run_report(parse_pairs(a.agent), a.out, a.baseline, workers=a.workers)
    else:
        print(budget(a.data, a.rl_steps))
