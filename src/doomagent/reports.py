import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


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
