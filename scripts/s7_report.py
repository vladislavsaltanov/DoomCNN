import json
import os

from doomagent.reports import plot_ppo_curves

names = ['ppo_ft', 'ppo_ft_flip', 'ppo_scratch_rl', 'ppo_scratch_total']  # ppo_ft_flip: S7 shaping
logs = {n: f'runs/{n}/val_log.json' for n in names if os.path.exists(f'runs/{n}/val_log.json')}
bc_level = None
if 'ppo_ft' in logs:
    with open(logs['ppo_ft']) as fh:
        bc_level = json.load(fh)[0]['val_score']  # evaluated before any PPO update
os.makedirs('reports', exist_ok=True)
plot_ppo_curves(logs, 'reports/s7_ppo.png', bc_level)
print(sorted(logs), 'imitation level:', bc_level)
