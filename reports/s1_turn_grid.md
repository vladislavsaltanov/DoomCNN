# S1: turn mode and aim margin, first 10 val seeds (base graphics)

| teacher | reward mean [95% CI] | kills/shot | shots | idle rate | length |
| --- | --- | --- | --- | --- | --- |
| binary, margin 0.2 (default) | 3.70 [2.50, 4.90] | 1.000 | 4.7 | 0.00 | 116 |
| binary, margin 0.1 | 3.80 [2.70, 4.90] | 1.000 | 4.8 | 0.00 | 114 |
| binary, margin 0.0 | 4.30 [3.10, 5.40] | 1.000 | 5.3 | 0.00 | 118 |
| delta 3 deg, margin 0.1 | 0.40 [-0.10, 1.00] | 1.000 | 1.4 | 0.10 | 85 |
| delta 3 deg, margin 0.0 | 0.20 [-0.20, 0.70] | 1.000 | 1.2 | 0.10 | 82 |
| delta 4 deg, margin 0.2 | 0.90 [0.50, 1.30] | 1.000 | 1.9 | 0.00 | 96 |
| delta 4 deg, margin 0.1 | 1.00 [0.50, 1.50] | 1.000 | 2.0 | 0.00 | 96 |
| delta 4 deg, margin 0.0 | 1.20 [0.50, 2.00] | 1.000 | 2.2 | 0.00 | 97 |
| delta 5 deg, margin 0.1 | 0.90 [0.30, 1.60] | 1.000 | 1.9 | 0.00 | 94 |

# S1: enemy box width vs turn step (725 enemy-box samples, median 6.5 px at 160 px width)

| turn step | narrowest box that can hold the crosshair | share of enemy boxes at least that wide |
| --- | --- | --- |
| binary, 14.06 deg | 20.0 px | 8.1% |
| delta 8 deg | 11.2 px | 16.1% |
| delta 4 deg | 5.6 px | 78.2% |
