"""E5 figure: the start model and the DAgger round models on the same 60 fresh matches (reports/dm/e5_rounds.png).

    python scripts/dm/dm_e5_report.py
"""
import json

from doomagent.dm.reports import models_figure

REPORTS = ['reports/dm/e4_fresh.json', 'reports/dm/e5_fresh_aux.json', 'reports/dm/e5_fresh_sp.json',
           'reports/dm/e5_fresh_sp_w5.json']
MODELS = ['dm3_m1_aux', 'dagger_aux_r1', 'dagger_sp_r2', 'dagger_sp_w5_r3', 'dagger_sp_w5_r4']

if __name__ == '__main__':
    seen = {}
    for p in REPORTS:
        with open(p) as fh:
            for n, s in json.load(fh)['models'].items():
                seen.setdefault(n, s)
    models_figure({n: seen[n] for n in MODELS}, 'reports/dm/e5_rounds.png', teacher=seen['teacher'])
