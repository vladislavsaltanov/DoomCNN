# S1: five actions, fine turn size, fine zone and aim margin (first 10 val seeds, base graphics)

| teacher | reward mean [95% CI] | kills/shot | shots | idle rate | length |
| --- | --- | --- | --- | --- | --- |
| binary, fine 4 deg, zone 0.15, margin 0.2 | 21.50 [19.00, 23.30] | 0.926 | 24.3 | 0.00 | 299 |
| binary, fine 2 deg, zone 0.15, margin 0.2 | 22.60 [21.70, 23.40] | 0.915 | 25.8 | 0.00 | 306 |
| binary, fine 1 deg, zone 0.15, margin 0.2 | 20.40 [17.70, 22.70] | 0.922 | 23.2 | 0.00 | 288 |
| delta 16 deg, fine 4 deg, zone 0.15, margin 0.2 | 16.50 [13.70, 19.00] | 0.781 | 22.4 | 0.00 | 251 |
| delta 16 deg, fine 2 deg, zone 0.15, margin 0.2 | 17.30 [15.60, 19.00] | 0.726 | 25.2 | 0.00 | 250 |
| binary, fine 2 deg, zone 0.10, margin 0.2 | 20.70 [17.10, 23.60] | 0.935 | 23.2 | 0.00 | 292 |
| binary, fine 2 deg, zone 0.25, margin 0.2 | 21.00 [17.60, 23.60] | 0.944 | 23.2 | 0.00 | 290 |
| binary, fine 2 deg, zone 0.15, margin 0.0 | 22.80 [22.10, 23.50] | 0.915 | 26.0 | 0.00 | 285 |
| delta 16 deg, fine 2 deg, zone 0.15, margin 0.0 | 19.10 [17.40, 20.90] | 0.773 | 26.0 | 0.00 | 280 |
| delta 16 deg, fine 2 deg, zone 0.25, margin 0.2 | 16.40 [13.60, 18.60] | 0.740 | 23.5 | 0.00 | 240 |

# S1: enemy box width vs turn step (3760 enemy-box samples, median 5.2 px at 160 px width)

| turn step | narrowest box that can hold the crosshair | share of enemy boxes at least that wide |
| --- | --- | --- |
| coarse binary, 14.06 deg | 20.0 px | 9.2% |
| coarse delta, 16 deg | 22.9 px | 7.9% |
| fine, 4 deg | 5.6 px | 44.8% |
| fine, 2 deg | 2.8 px | 98.4% |
| fine, 1 deg | 1.4 px | 99.7% |
