# E1: teacher improvement cycle

Each row: one change against the accepted teacher; 20 val matches, then 20 fresh matches (FRESH_SEEDS), paired by seed. A change is accepted only if the lower bound of the fresh difference is above 0 (plan v2, decision 9). Stop after two rejected changes in a row.

| change | settings | why | val frags (accepted → candidate) | fresh frags (accepted → candidate) | fresh difference [95% CI] | best bot (cand.) | K/D (cand.) | first of 8 (cand.) | accepted |
|---|---|---|---|---|---|---|---|---|---|
| hitscan_tol_3 | `hitscan_tol=3.0` | not aimed 69% of bot-on-screen steps; bullets spread wider than ±1° | 6.25 → 8.20 | 6.85 → 6.53 | -0.33 [-1.68, +1.10] | 8.45 | 3.07 | 35% | no |
| track_ahead | `track_ahead=True` | bots run ~2° a step sideways; the turn ends behind them | 6.25 → 6.70 | 6.85 → 6.70 | -0.15 [-2.08, +1.57] | 8.28 | 3.12 | 32% | no |
| barrel_check | `barrel_check=True` | 24% of deaths are suicides, also with the super shotgun: barrels next to the player | 6.25 → 6.60 | 6.85 → 6.83 | -0.03 [-0.17, +0.10] | 8.43 | 3.25 | 35% | no |
