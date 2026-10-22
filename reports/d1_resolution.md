# D1: is 160x120 enough? (teacher on the val matches)

Command: `python scripts/dm_resolution.py`

| no shots at bots narrower than (px at 160) | frags [95% CI] | vs no limit [95% CI] | loss |
| --- | --- | --- | --- |
| 0 | 4.05 [2.80, 5.35] | +0.00 [+0.00, +0.00] | -0% |
| 2 | 4.75 [3.60, 5.90] | +0.70 [+0.10, +1.40] | -17% |
| 3 | 4.20 [3.05, 5.40] | +0.15 [-1.10, +1.60] | -4% |
| 4 | 3.45 [2.40, 4.55] | -0.60 [-1.65, +0.55] | 15% |
| 6 | 2.25 [1.55, 2.95] | -1.80 [-3.20, -0.35] | 44% |

Rockets fired: 411; bot width at the shot: median 4.2 px, 43% narrower than 4 px.

Verdict: forbidding shots under 4 px costs 15% of the frags: within 15%, 160x120 stays.
