# Extended-metrics selection-slice scores for the original lr 1e-3 checkpoints

Scored on 2026-10-14 by `eval_legacy_selection.py`. No training. Weights were read from `checkpoints/` and not modified. The test split was not read. Sample seed 42, the same seed the original checkpoint loop used. Real selection rows were checked first and had 0 hard-validity violations for every seed and both scalers. An ineligible file is the last epoch, not a selected model.

Protocol B and the utility ratio use the real test split. They are implemented and covered by fixture tests. They were not run on Adult here, so this table could not be used to rank anything with the test set. The published test detection and efficacy numbers stay the 2026-10-13 notebook figures.

Hard validity counts are rows, out of 5,210 selection rows. Detection is protocol A (4-fold forest, 80 trees, depth 16, seed 42) on decoded original-domain rows.

| Checkpoint | Accepted | Epoch | Integer rows | Domain rows | Negative capital | KS mean | TV mean | Coverage | Rare recall | AUC | Exact copies |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gan_linear seed 42 | no |  | 5210 | 3855 | 3798 | 0.5943 | 0.3293 | 0.5634 | 0.2143 | 0.999999 | 0 |
| gan_linear seed 68 | no |  | 5210 | 4570 | 4568 | 0.5641 | 0.4060 | 0.3786 | 0.0930 | 1.000000 | 0 |
| gan_linear seed 124 | no |  | 5210 | 3575 | 3575 | 0.5191 | 0.3184 | 0.5016 | 0.0465 | 1.000000 | 0 |
| gan_sigmoid seed 42 | yes | 110 | 5210 | 0 | 0 | 0.4726 | 0.3142 | 0.3544 | 0.0000 | 0.999996 | 0 |
| gan_sigmoid seed 68 | yes | 90 | 5210 | 0 | 0 | 0.5800 | 0.3759 | 0.2346 | 0.0233 | 0.999994 | 0 |
| gan_sigmoid seed 124 | yes | 20 | 5210 | 0 | 0 | 0.4980 | 0.3358 | 0.3079 | 0.0000 | 1.000000 | 0 |
| cgan_linear seed 42 | no |  | 5210 | 4169 | 4052 | 0.3115 | 0.2462 | 0.5206 | 0.1905 | 1.000000 | 0 |
| cgan_linear seed 68 | no |  | 5210 | 3414 | 3401 | 0.3677 | 0.3834 | 0.5531 | 0.1860 | 0.999991 | 0 |
| cgan_linear seed 124 | no |  | 5210 | 2480 | 2399 | 0.3703 | 0.2944 | 0.6459 | 0.2093 | 1.000000 | 0 |
| cgan_sigmoid seed 42 | yes | 70 | 5210 | 0 | 0 | 0.4506 | 0.3809 | 0.2387 | 0.0000 | 0.999974 | 0 |
| cgan_sigmoid seed 68 | no |  | 2020 | 0 | 0 | 0.9984 | 0.8108 | 0.3004 | 0.0930 | 1.000000 | 0 |
| cgan_sigmoid seed 124 | yes | 10 | 5210 | 0 | 0 | 0.5772 | 0.4075 | 0.2174 | 0.0000 | 1.000000 | 0 |

`cgan_sigmoid` seed 68 also has synthetic duplicate rate 0.7948 and a nearest-neighbor ratio of 10.49. The other runs have duplicate rate 0. Accepted sigmoid runs have nearest-neighbor ratios 0.28 to 0.51 and zero exact copies of fit rows. Those neighbor ratios are a privacy proxy only.

No run meets the planning goals. Every run fails hard validity. Protocol A AUC is saturated near 1. Category coverage on the accepted sigmoid runs is about 0.22 to 0.35, and rare-category recall is 0 on four of the five accepted runs. Capital zero-rate gaps remain large. This is not a successful fidelity result.
