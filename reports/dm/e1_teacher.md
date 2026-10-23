# E1: cheater teacher on the val matches (MAP02)

Command: `python scripts/dm/dm_teacher_report.py --diag runs/dm/cycle/ac3b6a5fc1_val20.json`

Matches: 20 (base graphics, VAL_SEEDS)

| player | frags [95% CI] | deaths | K/D (mean, total) | suicides | hits per shot | items | path | stuck | sees a bot | median bot's frags [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| teacher | 7.85 [6.55, 9.40] | 2.6 | 4.15, 3.02 | 0.3 | 0.61 | 5.8 | 19561 | 0.1% | 64% | 3.35 [2.75, 3.95] |
| random | -0.20 [-0.40, -0.05] | 1.6 | -0.11, -0.12 | 0.2 | 0.04 | 2.9 | 13139 | 5.2% | 16% | 3.90 [3.40, 4.45] |

Best bot of the match: 7.90 [7.10, 8.65] frags; first of 8 in 55 % of matches.

## Gate A v2.1 (revised 2026-10-23)

- PASS: K/D over all matches >= 3: 3.019230769230769
- PASS: first of 8 in >= 50 % of matches: 0.55
- PASS: frags - best bot, upper 95% bound >= 0: (-0.05, -1.3, 1.35)

## Teacher decisions (share of steps)

combat 61%, chase 11%, gear 11%, items 3%, patrol 15%

## Weapon in hand (share of steps)

fist 5%, pistol 22%, super shotgun 13%, chaingun 45%, rocket launcher 14%, plasma rifle 1%, BFG9000 0%

## Diagnostics (runs/dm/cycle/ac3b6a5fc1_val20.json)

Deaths by cause: suicide 13%, bot seen 87%, unseen 0%

Hits per shot by weapon: pistol 0.76 (207 shots), super shotgun 0.80 (104 shots), chaingun 0.75 (2304 shots), rocket launcher 0.43 (91 shots), plasma rifle 0.00 (14 shots), BFG9000 1.00 (3 shots)

Bot on screen, by the reason not to fire: shoot 17%, switching 3%, not_aimed 66%, unsafe 0%, no_ammo 0%, reloading 8%, no_combat 6%

Teacher frags per match: 5 5 6 4 7 3 10 9 8 15 9 8 5 13 5 5 15 7 9 9

Best bot per match: 6 4 8 6 11 8 10 8 7 8 9 4 8 9 7 9 11 8 9 8

![teacher](e1_teacher.png)
