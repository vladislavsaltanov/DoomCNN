# D3: where the teacher loses (scripts/dm_teacher_diag.py)

Each row: the teacher with some upgrades switched on (names are the --set values; x_ is the D1 teacher). val = VAL_SEEDS, fresh = FRESH_SEEDS (not used for tuning).

| run | matches | frags [95% CI] | deaths | 1st of 8 | suicide share | hit share | not aimed (bot on screen) | wall stop |
|---|---|---|---|---|---|---|---|---|
| aim | 12 | 6.67 [5.33, 7.75] | 4.50 | 0.83 | 0.17 | 0.40 | 0.43 | 0.01 |
| base | 20 | 6.25 [4.80, 7.70] | 3.60 | 0.80 | 0.12 | 0.44 | 0.59 | 0.15 |
| fresh_x_ | 20 | 6.45 [5.00, 8.00] | 3.90 | 0.90 | 0.14 | 0.41 | 0.61 | 0.07 |
| fresh_x_shot_check-rocket_aim_slack-24 | 20 | 6.70 [5.45, 7.95] | 3.75 | 0.90 | 0.19 | 0.41 | 0.39 | 0.02 |
| shot_check-rocket | 20 | 6.40 [5.05, 7.80] | 4.30 | 0.85 | 0.20 | 0.41 | 0.64 | 0.01 |
| shot_check-rocket_aim_slack-24 | 20 | 7.50 [6.40, 8.50] | 4.35 | 0.90 | 0.16 | 0.42 | 0.43 | 0.02 |
| shot_check-rocket_aim_slack-24_fine_turn_delta-1.0 | 20 | 5.40 [4.40, 6.40] | 4.00 | 0.70 | 0.11 | 0.35 | 0.53 | 0.01 |
| shot_check-rocket_aim_slack-24_fine_turn_delta-1.5 | 20 | 5.30 [4.25, 6.20] | 4.15 | 0.85 | 0.16 | 0.34 | 0.57 | 0.02 |
| shot_check-rocket_aim_slack-24_fine_turn_delta-1.5_dodge-True | 20 | 5.60 [4.25, 7.10] | 4.75 | 0.65 | 0.28 | 0.42 | 0.59 | 0.02 |
| shot_check-rocket_aim_slack-24_fine_turn_delta-2.0 | 20 | 4.15 [2.80, 5.65] | 4.50 | 0.65 | 0.29 | 0.32 | 0.60 | 0.01 |
| shot_check-rocket_dodge-True | 20 | 6.80 [5.50, 8.15] | 4.05 | 0.90 | 0.28 | 0.44 | 0.75 | 0.01 |

Fresh seeds, paired by seed: rocket ray + aim_slack 24 minus D1 teacher = +0.25 [-1.65, +2.10] frags (no measurable gain; the +1.25 on val was tuning noise).
Not-aimed steps (12 val matches, rocket + slack 24): aim error median 0.244 of the half screen (about 14 degrees), target half-width 0.02 (about 1 degree), target at about 790 units, lead jump 0.006; 65% of them outside the fine-turn zone. The limit is the turn set (14 and 2 degrees per step), not the lead.
MAP02 probe (cig.wad, 7 bots, seed 35000, random player): the player starts with a pistol and 50 bullets; on the map 5 chainguns, 3 super shotguns, 2 rocket launchers, 2 plasma rifles, 1 BFG and their ammo, 179 sectors; bots scored 2-6 frags each in 2 minutes (MAP01 median bot about 0.65).

