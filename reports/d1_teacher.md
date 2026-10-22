# D1: teacher on the val matches

Command: `python scripts/dm_teacher_report.py --set safe_dist=128 --set chase_s=1 --set aim_margin=0`

Matches: 20 (base x20)

| player | frags [95% CI] | deaths | K/D (mean, total) | suicides | hits per rocket | items | path | stuck | sees a bot | median bot's frags [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| teacher | 6.25 [4.80, 7.70] | 3.6 | 2.35, 1.74 | 0.4 | 0.45 | 0.0 | 10806 | 0.7% | 53% | 0.65 [0.10, 1.20] |
| random | -15.30 [-16.60, -13.90] | 18.3 | -0.83, -0.84 | 16.5 | 0.13 | 0.0 | 10597 | 1.5% | 20% | 0.35 [-0.05, 0.80] |

## Gate A

- PASS: K/D over all matches > 1: 1.7361111111111112
- PASS: frags >= median bot's frags: (6.25, 0.65)
- PASS: stuck share < 5 %: 0.006717599527895047

## Teacher decisions (share of steps)

combat 51%, chase 10%, items 12%, patrol 27%

Teacher frags per match: 4 10 3 4 13 10 6 10 5 1 3 6 5 6 7 9 11 4 7 1

Teacher deaths per match: 4 5 0 4 1 2 4 4 3 4 3 4 5 4 5 2 7 2 5 4
