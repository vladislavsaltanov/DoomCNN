# S1: five actions, fine turn size, fine zone and aim margin (first 10 val seeds, base graphics)

| teacher | reward mean [95% CI] | kills/shot | shots | idle rate | length |
| --- | --- | --- | --- | --- | --- |
| binary, fine 4 deg, zone 0.15, margin 0.2 | 4.20 [3.20, 5.20] | 0.981 | 5.3 | 0.00 | 114 |
| binary, fine 2 deg, zone 0.15, margin 0.2 | 6.20 [5.40, 7.10] | 1.000 | 7.2 | 0.00 | 119 |
| binary, fine 1 deg, zone 0.15, margin 0.2 | 5.30 [4.90, 5.70] | 1.000 | 6.2 | 0.00 | 118 |
| delta 16 deg, fine 4 deg, zone 0.15, margin 0.2 | 5.80 [4.80, 6.90] | 0.932 | 7.3 | 0.00 | 119 |
| delta 16 deg, fine 2 deg, zone 0.15, margin 0.2 | 6.10 [5.50, 6.70] | 0.986 | 7.2 | 0.00 | 119 |
| binary, fine 2 deg, zone 0.10, margin 0.2 | 6.00 [5.30, 6.70] | 1.000 | 7.0 | 0.00 | 123 |
| binary, fine 2 deg, zone 0.25, margin 0.2 | 5.40 [4.70, 6.20] | 1.000 | 6.4 | 0.00 | 116 |
| binary, fine 2 deg, zone 0.15, margin 0.0 | 5.20 [4.70, 5.70] | 0.968 | 6.3 | 0.00 | 112 |
| delta 16 deg, fine 2 deg, zone 0.15, margin 0.0 | 6.10 [5.40, 6.80] | 0.973 | 7.3 | 0.00 | 115 |
| delta 16 deg, fine 2 deg, zone 0.25, margin 0.2 | 5.70 [4.70, 6.70] | 0.971 | 6.8 | 0.00 | 117 |

# S1: enemy box width vs turn step (513 enemy-box samples, median 6.0 px at 160 px width)

| turn step | narrowest box that can hold the crosshair | share of enemy boxes at least that wide |
| --- | --- | --- |
| coarse binary, 14.06 deg | 20.0 px | 2.5% |
| coarse delta, 16 deg | 22.9 px | 1.0% |
| fine, 4 deg | 5.6 px | 61.0% |
| fine, 2 deg | 2.8 px | 99.6% |
| fine, 1 deg | 1.4 px | 100.0% |
