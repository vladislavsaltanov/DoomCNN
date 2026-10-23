# D1: teacher on the val matches

Command: `python scripts/dm_teacher_report.py --out reports/d1_teacher_defaults`

Matches: 20 (base x20)

| player | frags [95% CI] | deaths | K/D (mean, total) | suicides | hits per rocket | items | path | stuck | sees a bot | median bot's frags [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| teacher | 4.05 [2.80, 5.35] | 4.0 | 1.08, 1.01 | 0.7 | 0.52 | 0.0 | 11191 | 0.3% | 52% | 0.40 [-0.15, 0.95] |
| random | -15.30 [-16.60, -13.90] | 18.3 | -0.83, -0.84 | 16.5 | 0.13 | 0.0 | 10597 | 1.5% | 20% | 0.35 [-0.05, 0.80] |

## Gate A

- PASS: K/D over all matches > 1: 1.0125
- PASS: frags >= median bot's frags: (4.05, 0.4)
- PASS: stuck share < 5 %: 0.003336147805165917

## Teacher decisions (share of steps)

combat 52%, chase 16%, items 6%, patrol 27%

Teacher frags per match: 5 3 1 4 5 7 5 0 2 0 6 4 11 1 5 5 10 2 3 2

Teacher deaths per match: 4 5 2 3 8 3 3 3 5 1 6 4 3 4 4 3 4 7 4 4
