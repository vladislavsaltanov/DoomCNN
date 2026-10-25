# D1: teacher parameter grid (10 val matches each)

Command: `python scripts/dm_teacher_grid.py`

Defaults: frags 3.20, K/D total 0.86

| parameter | value | frags [95% CI] | vs default [95% CI] | K/D total | rule |
| --- | --- | --- | --- | --- | --- |
| safe_dist | 128.0 | 6.20 [5.30, 6.90] | +3.00 [+1.80, +4.20] | 1.68 | better |
| safe_dist | 224.0 | 4.10 [2.60, 5.60] | +0.90 [-0.50, +2.80] | 1.24 | keep default |
| chase_s | 1.0 | 6.40 [5.20, 7.90] | +3.20 [+1.10, +5.30] | 1.94 | better |
| chase_s | 3.0 | 5.80 [4.00, 7.40] | +2.60 [+0.40, +4.80] | 1.35 | better |
| low_health | 30.0 | 3.50 [2.10, 4.90] | +0.30 [+0.00, +0.70] | 0.92 | keep default |
| low_health | 70.0 | 3.00 [1.60, 4.50] | -0.20 [-0.50, +0.00] | 0.83 | keep default |
| low_ammo | 3.0 | 3.30 [1.90, 4.60] | +0.10 [+0.00, +0.30] | 0.97 | keep default |
| low_ammo | 8.0 | 3.70 [1.90, 5.70] | +0.50 [+0.00, +1.50] | 1.00 | keep default |
| aim_margin | 0.0 | 6.30 [4.30, 8.00] | +3.10 [+1.10, +5.10] | 1.80 | better |
| aim_margin | 0.35 | 4.40 [2.90, 6.00] | +1.20 [+0.00, +2.50] | 1.52 | keep default |

Chosen: safe_dist=128.0, chase_s=1.0, aim_margin=0.0
