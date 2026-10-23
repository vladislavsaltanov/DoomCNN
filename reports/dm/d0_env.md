# D0: deathmatch environment on cig.wad MAP01

## Label names (random agent, one 2-minute match, label-steps)

| name | class | label-steps |
| --- | --- | --- |
| DeadDoomPlayer | other | 1455 |
| RocketSmokeTrail | other | 1396 |
| Medikit | health | 1125 |
| Rocket | other | 537 |
| RocketAmmo | ammo | 291 |
| DoomPlayer | bot | 263 |
| ExplosiveBarrel | other | 212 |
| TeleportFog | other | 146 |
| GreenArmor | armor | 76 |
| BlueArmor | armor | 72 |
| DeadExplosiveBarrel | other | 21 |
| ItemFog | other | 16 |

## Physics per agent step (4 tics), no bots

| action | per step |
| --- | --- |
| forward, held (units) | 7.1 15.6 21.4 25.3 27.9 29.7 30.9 31.7 32.2 32.6 32.8 33.0 |
| strafe left, held (units) | 4.3 10.2 16.8 21.6 25.0 27.2 28.8 21.6 21.8 22.3 24.5 26.1 |
| turn left, hold (degrees) | 7.03 12.30 14.06 14.06 14.06 14.06 14.06 14.06 14.06 14.06 14.06 14.06 |
| turn left, alternate (degrees) | 7.03 -12.30 14.06 -14.06 14.06 -14.06 14.06 -14.06 14.06 -14.06 14.06 -14.06 |
| fine left, hold (degrees) | 2.02 2.02 2.02 2.02 2.02 2.02 2.02 2.02 2.02 2.02 2.02 2.02 |
| fine left, alternate (degrees) | 2.02 -2.00 2.02 -2.00 2.02 -2.00 2.02 -2.00 2.02 -2.00 2.02 -2.00 |
| forward left, hold (degrees) | 7.03 12.30 14.06 14.06 14.06 14.06 14.06 14.06 14.06 14.06 14.06 14.06 |
| forward left, alternate (degrees) | 7.03 -12.30 14.06 -14.06 14.06 -14.06 14.06 -14.06 14.06 -14.06 14.06 -14.06 |
| forward left, held (units) | 7.1 15.6 21.1 24.6 26.7 27.9 28.5 6.3 0.8 5.6 10.4 14.7 |

## Number of bots (random agent, 5 val matches each)

| bots | ms per step | median bot frags [95% CI] | mean bot frags | bots below 0 | all frags per match | random agent frags | random suicides | random sees a bot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 7.6 | 0.60 [0.20, 1.00] | 0.53 | 13% | -15.6 | -17.2 | 17.2 | 10% |
| 5 | 7.7 | 0.40 [0.00, 0.80] | 0.52 | 12% | -12.8 | -15.4 | 16.0 | 18% |
| 7 | 8.1 | 0.00 [-1.00, 0.80] | 0.34 | 40% | -11.8 | -14.2 | 15.4 | 21% |

## Items and map

- objects at match start and end (objects_info): BlueArmor 1/1, DeadDoomPlayer 2/32, DoomPlayer 7/7, ExplosiveBarrel 13/2, GreenArmor 2/2, Medikit 28/28, Rocket 0/1, RocketAmmo 21/21, RocketBox 1/1, RocketSmokeTrail 0/2, TeleportFog 10/0
- items the random agent picked up: 0
- MAP01 line specials: {"512": 1} (no doors or lifts: no USE button)

## Repeatability

- two matches with one seed and one action sequence (120 steps): identical
- distinct matches among the 20 VAL_SEEDS: 20
