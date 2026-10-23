# E1: teacher improvement cycle

Each row: one change against the accepted teacher; 20 val matches, then 20 fresh matches (FRESH_SEEDS), paired by seed. A change is accepted only if the lower bound of the fresh difference is above 0 (plan v2, decision 9). Stop after two rejected changes in a row.

| change | settings | why | val frags (accepted → candidate) | fresh frags (accepted → candidate) | fresh difference [95% CI] | best bot (cand.) | K/D (cand.) | first of 8 (cand.) | accepted |
|---|---|---|---|---|---|---|---|---|---|
| hitscan_tol_3 | `hitscan_tol=3.0` | not aimed 69% of bot-on-screen steps; bullets spread wider than ±1° | 6.25 → 8.20 | 6.85 → 6.53 | -0.33 [-1.68, +1.10] | 8.45 | 3.07 | 35% | no |
| track_ahead | `track_ahead=True` | bots run ~2° a step sideways; the turn ends behind them | 6.25 → 6.70 | 6.85 → 6.70 | -0.15 [-2.08, +1.57] | 8.28 | 3.12 | 32% | no |
| barrel_check | `barrel_check=True` | 24% of deaths are suicides, also with the super shotgun: barrels next to the player | 6.25 → 6.60 | 6.85 → 6.83 | -0.03 [-0.17, +0.10] | 8.43 | 3.25 | 35% | no |
| weapon_first_256 | `weapon_first=256.0` | pistol 45% of steps: fights with a pistol keep it from gearing up | 6.25 → 7.85 | 6.85 → 8.25 | +1.40 [+0.25, +2.60] | 8.47 | 2.70 | 48% | yes |
| armor_50 | `weapon_first=256.0 armor_below=50.0` | 76% of deaths in fights; armor halves the damage | 7.85 → 7.50 | 8.25 → 8.53 | +0.28 [-1.38, +1.77] | 8.65 | 4.37 | 52% | no |
| chase_3 | `weapon_first=256.0 chase_s=3.0` | bots get away; a 1 s chase loses them | 7.85 → 8.30 | 8.25 → 7.75 | -0.50 [-2.10, +0.95] | 7.65 | 2.42 | 52% | no |
| min_vis_1 | `weapon_first=256.0 min_vis_px=1.0` | 36% of bot labels have 0 pixels (behind walls and ledges); shots under 50 px hit 13% | 7.85 → 8.00 | 8.25 → 8.47 | +0.23 [-1.02, +1.40] | 7.97 | 3.36 | 62% | no |
| min_vis_50 | `weapon_first=256.0 min_vis_px=50.0` | 36% of bot labels have 0 pixels (behind walls and ledges); shots under 50 px hit 13% | 7.85 → 9.60 | 8.25 → 9.90 | +1.65 [-0.05, +3.15] | 8.68 | 3.84 | 70% | no |
| min_vis_200 | `weapon_first=256.0 min_vis_px=200.0` | 36% of bot labels have 0 pixels (behind walls and ledges); shots under 50 px hit 13% | 7.85 → 8.15 | 8.25 → 8.43 | +0.17 [-1.50, +1.78] | 7.85 | 2.67 | 55% | no |
