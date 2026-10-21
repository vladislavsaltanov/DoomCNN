# S5: R1 (teacher on train variants + augmentations) against R0

val plan: base graphics 30 seeds, 8 val variants x 10 seeds. Teacher on base: 22.67 [21.90, 23.23].

| model | val_score | base | ood_mean | tex_val 00 / 01 / 02 / 03 | spr_val cpos_skel / skel_cpos | mix_val 00 / 01 |
| --- | --- | --- | --- | --- | --- | --- |
| bc0 | 5.02 | 22.90 | 2.79 | -1.0 / -0.4 / -0.6 / 0.8 | 11.2 / 6.4 | 4.9 / 1.0 |
| bc0_aug | 9.93 | 21.33 | 8.50 | 2.1 / 0.3 / 3.9 / 1.6 | 18.3 / 22.3 | 16.8 / 2.7 |
| r1 | 14.00 | 22.73 | 12.91 | 15.9 / 9.7 / 16.0 / 4.8 | 24.1 / 20.0 | 4.8 / 8.0 |

r1 val_score by checkpoint (epochs 2..12): 10.16, 9.81, 10.60, 13.32, 11.80, 14.00.
