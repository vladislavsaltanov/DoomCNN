"""E6 curves of a PPO run (reports/dm/e6_ppo/e6_<run>.png) from its log.jsonl; works while the run goes on.

    python scripts/dm/dm_e6_report.py runs/dm/ppo_ks
"""
import argparse
import json
import os

from doomagent.dm.reports import ppo_figure

TEACHER_VAL = 11.90  # the final teacher on val (Gate A v2.1 recount)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--out')
    a = ap.parse_args()
    with open(os.path.join(a.run, 'log.jsonl')) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    out = a.out or f'reports/dm/e6_ppo/e6_{os.path.basename(os.path.normpath(a.run))}.png'
    ppo_figure(rows, out, TEACHER_VAL)
    print(out)
